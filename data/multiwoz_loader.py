from datasets import Dataset
from transformers import GPT2Tokenizer
from torch.utils.data import Sampler
import json
import re
from random import shuffle
from collections import OrderedDict
from database.database import MultiWOZDatabase
import torch
from torch.nn.utils.rnn import pad_sequence




added_tokens = ["<|system|>", "<|user|>", "<|belief|>", "<|database|>"]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", additional_special_tokens = added_tokens)
db = MultiWOZDatabase()


def delexicalize_utterance(row):
    utterance = row['utterance']
    slot_details = row['slot_details']
    if len(slot_details['span_start']) == 0:
        return {"delex_utterance": utterance}
    delexicalized_utterance = ""
    for i in range(len(slot_details['span_start'])):
        if i ==0:
            first_part_of_string = utterance[:slot_details['span_start'][i]]
        else:
            first_part_of_string = utterance[slot_details['span_end'][i-1]:slot_details['span_start'][i]]      
        
        
        if i == len(slot_details['span_start'])-1:
            last_part_of_string = utterance[slot_details['span_end'][i]:]
        else:
            last_part_of_string = ""
        
        delexicalized_utterance = ''.join((delexicalized_utterance, first_part_of_string, f"[{slot_details['act_slot_name'][i]}]", last_part_of_string))

    return {"delex_utterance": delexicalized_utterance}

def get_belief_state_and_db_results(row):
    frame = row['belief_state_details']
    belief_state = {}
    for i,service in enumerate(frame['service']):
        zipped_slots =  list(zip(frame['state'][i]['slots_values']['slots_values_name'], frame['state'][i]['slots_values']['slots_values_list']))
        belief_state[service] =  {i.split('-')[1]:j[0] for i,j in zipped_slots}
    db_results = {}
    for domain,constraints in belief_state.items():
        db_results[domain] = len(db.query(domain, constraints = constraints))
    return {"belief_state": json.dumps(belief_state), "database_results" : json.dumps(db_results)}


def create_utterances_dataset(raw_dataset_dict, split = "train", k = 3):
    dataset = {}
    dataset['utterance'] = []
    dataset['context'] = []
    dataset['slot_details'] = []
    dataset['belief_state_details'] = []
    # for i in range(100):
    for i in range(len(raw_dataset_dict[split])):
        turns = raw_dataset_dict[split][i]['turns']
        system_utterance_indexes = [i for i,j in enumerate(turns['speaker']) if j==1]
        dataset['utterance'] += [turns['utterance'][i] for i in system_utterance_indexes]
        dataset['context'].extend([turns['utterance'][max(0, i-k):i] for i in system_utterance_indexes])
        dataset['slot_details'] += [turns['dialogue_acts'][i]['span_info'] for i in system_utterance_indexes]
        dataset['belief_state_details'] += [turns['frames'][max(0,i-1)] for i in system_utterance_indexes]

    utterances_dataset = Dataset.from_dict(dataset)
    utterances_dataset = utterances_dataset.map(delexicalize_utterance, remove_columns=["slot_details"])
    utterances_dataset = utterances_dataset.map(get_belief_state_and_db_results, remove_columns=["belief_state_details"])
    return utterances_dataset

class BucketBatchSampler(Sampler):

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        # For each data item we store the index and the length of utterance + context.
        self.ind_n_len = []
        for i in range(len(data)):
            self.ind_n_len.append((i, (len(data[i]['utterance']) + sum([len(turn) for turn in data[i]['context']]) + len(data[i]['belief_state']))))

        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)


    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # batch_map is a dictionary with length as key and the indices as value.
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        # shuffle all the batches so they aren't ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i

def prepare_belief_state_for_input(bs):
    """
    Convert the belief state to an appropriate format to be fed into the model. Removes quotes and some brackets, adds spaces before`:` , etc.
    """
    bs = re.sub('[\[\]"]', '', bs)
    return re.sub('(?<=[a-zA-Z]):', ' :', bs)

def collator(batch):
    utterances_batch, contexts_batch, belief_states_batch,  delex_utterances_batch, database_results_batch = [], [], [], [], [] 
    for example in batch:
        utterances_batch.append(example['utterance'])
        contexts_batch.append(example['context'])
        belief_states_batch.append(example['belief_state'])
        delex_utterances_batch.append(example['delex_utterance'])
        database_results_batch.append(example['database_results'])
    
    contexts_batch_old = contexts_batch
    for i in range(len(contexts_batch)):
        turns = [''.join(('<|user|> ', turn)) if j%2==0 else ''.join(('<|system|> ', turn)) for j,turn in enumerate(contexts_batch[i][::-1])][::-1]
        # Flattening the contexts.
        contexts_batch[i] = [item for sublist in tokenizer(turns)['input_ids'] for item in sublist]

    
    utterances_batch = tokenizer(utterances_batch)['input_ids']

    delex_utterances_batch = [''.join((delex_utterance, '<|endoftext|>')) for delex_utterance in delex_utterances_batch]
    delex_utterances_batch = tokenizer(delex_utterances_batch)['input_ids']
    database_results_batch = [''.join(('<|database|> ', re.sub('[\[\]{}"]', '', db_result), '<|endoftext|>')) for db_result in database_results_batch]
    database_results_batch = tokenizer(database_results_batch)['input_ids']
    belief_states_batch = [''.join(('<|belief|> ', prepare_belief_state_for_input(belief_state))) for belief_state in belief_states_batch]
    belief_states_batch = tokenizer(belief_states_batch)['input_ids']

    labels_batch = []
    context_mask_batch = []
    belief_mask_batch = []
    database_mask_batch = []
    utterance_mask_batch = []

    for context, belief, database_result, delex_utterance in zip(contexts_batch, belief_states_batch, database_results_batch, delex_utterances_batch):
        label_single = torch.tensor(context + belief + database_result + delex_utterance, dtype=torch.long)
        labels_batch.append(label_single)

        context_mask_batch.append(torch.tensor([True if i<len(context) else False for i in range(label_single.shape[0])]))
        belief_mask_batch.append(torch.tensor([True if (i>=len(context) and i<(len(context) + len(belief))) else False for i in range(label_single.shape[0])]))
        database_mask_batch.append(torch.tensor([True if (i>=(len(context) + len(belief)) and i<(len(context) + len(belief) + len(database_result))) else False for i in range(label_single.shape[0])]))
        utterance_mask_batch.append(torch.tensor([True if i>=(len(context) + len(belief) + len(database_result)) else False for i in range(label_single.shape[0])]))

    labels_batch = pad_sequence(labels_batch, batch_first = True)
    context_mask_batch = pad_sequence(context_mask_batch, batch_first = True)
    belief_mask_batch = pad_sequence(belief_mask_batch, batch_first = True)
    database_mask_batch = pad_sequence(database_mask_batch, batch_first = True)
    utterance_mask_batch = pad_sequence(utterance_mask_batch, batch_first = True)
    
    return {
        "labels" : labels_batch,
        "context_mask" : context_mask_batch,
        "belief_mask" : belief_mask_batch,
        "database_mask" : database_mask_batch,
        "utterance_mask" : utterance_mask_batch,

    }