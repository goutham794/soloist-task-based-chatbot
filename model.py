
from tqdm.auto import tqdm
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def eval_fn(data_loader, model):
    model.eval()
    batch_losses = []
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch_ids = batch['labels']
        batch_attention_mask = batch['context_mask'] + batch['belief_mask'] + batch['database_mask'] + batch['utterance_mask']

        dialog_state_targets = batch_ids * batch['belief_mask']
        utterance_targets = batch_ids * batch['utterance_mask']

        utterance_targets[batch['utterance_mask'] == 0] = -100
        dialog_state_targets[batch['belief_mask'] == 0] = -100
        with torch.no_grad():
            utterance_outputs = model(input_ids = batch_ids, 
                            attention_mask = batch_attention_mask,
                            labels = utterance_targets
                            )
            dialog_state_outputs = model(input_ids = batch_ids, 
                            attention_mask = batch_attention_mask,
                            labels = dialog_state_targets
                            )

        loss = utterance_outputs.loss + dialog_state_outputs.loss
        batch_losses.append(loss.item())
    return sum(batch_losses)/len(batch_losses)
    
def fit(num_epochs, model, opt, train_dl, valid_dl, optimizer, scheduler, num_training_steps):
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(num_epochs):
        for batch in train_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_ids = batch['labels']
            batch_attention_mask = batch['context_mask'] + batch['belief_mask'] + batch['database_mask'] + batch['utterance_mask']

            dialog_state_targets = batch_ids * batch['belief_mask']
            utterance_targets = batch_ids * batch['utterance_mask']

            utterance_targets[batch['utterance_mask'] == 0] = -100
            dialog_state_targets[batch['belief_mask'] == 0] = -100

            utterance_outputs = model(input_ids = batch_ids, 
                            attention_mask = batch_attention_mask,
                            labels = utterance_targets
                            )
            dialog_state_outputs = model(input_ids = batch_ids, 
                            attention_mask = batch_attention_mask,
                            labels = dialog_state_targets
                            )
            loss = utterance_outputs.loss + dialog_state_outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            # print('Train Loss: {:.4f} '.format(loss.item()))
            progress_bar.update(1)
        valid_loss = eval_fn(valid_dl, model)
        print('Validation loss {:.4f}'.format(loss.item(), valid_loss))

