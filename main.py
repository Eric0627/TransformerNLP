import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer, AdvancedTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import EncoderWithClassifier, Decoder, Decoder_MQA

import torch.nn as nn
import torch.optim as optim
from utilities import Utilities
import argparse
import matplotlib.pyplot as plt
import numpy as np

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 50  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs, _ = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        _, loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        if len(losses) >= eval_iters: break
    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)
    decoderLMmodel.train()
    return perplexity

def main():
    parser = argparse.ArgumentParser(description = "CSE 256 PA2")
    parser.add_argument("--part", type=int, help = "Run the code of part 1, 2, or 3 (int)", required = True)
    parser.add_argument("--visualization", type=bool, help = "Draw a plot of training statistics (bool)", required = False)
    args = parser.parse_args()

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)
    
    # for the classification task, you will train for a fixed number of epochs like this:
    if args.part == 1:
        # load data
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        if args.visualization:
            # draw the distribution plot
            train_count, test_count = [0, 0, 0], [0, 0, 0]
            for label, _ in train_CLS_dataset.samples:
                train_count[label] += 1
            for label, _ in test_CLS_dataset.samples:
                test_count[label] += 1
            print(train_count, test_count)
            labels = ['Barack Obama', 'George W. Bush', 'George H. Bush']
            fig = plt.figure()
            x_axis = np.arange(len(labels)) 
            plt.bar(x_axis - 0.2, train_count, 0.4, label = 'Train data') 
            plt.bar(x_axis + 0.2, test_count, 0.4, label = 'Test data') 
            plt.xticks(x_axis, labels) 
            plt.title('Speech segment count distribution')
            plt.xlabel('Politicians')
            plt.ylabel('Counts')
            plt.legend()
            for i, count in enumerate(train_count):
                plt.text(x_axis[i] - 0.2, count + 0.1, str(count), ha='center', va='bottom', color='black')

            for i, count in enumerate(test_count):
                plt.text(x_axis[i] + 0.2, count + 0.1, str(count), ha='center', va='bottom', color='black')
            plt.savefig('part1_data_count.png')
            # get the Jaccard similarity between classes in terms of top 100 tokens
            train_tokens = train_CLS_dataset.top_k_tokens(100)
            test_tokens = test_CLS_dataset.top_k_tokens(100)
            for i in range(3):
                for j in range(i, 3):
                    set1, set2 = set(train_tokens[i]), set(test_tokens[j])
                    intersection = set1.intersection(set2)
                    union = set1.union(set2)
                    jaccard_similarity = len(intersection) / len(union)
                    print(f'label {i}, label {j}, the Jaccard similarity {jaccard_similarity}')
        # instantiate the encoder with classifier model
        encoder_classifier = EncoderWithClassifier(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, device, n_hidden, n_output)
        print(f'number of parameters: {encoder_classifier.num_params()}')
        encoder_classifier.train()
        # create an optimizer
        optimizer = optim.AdamW(encoder_classifier.parameters(), lr=learning_rate)
        # create arrays of training statistics
        train_accuracy, train_loss, test_accuracy = [], [], []
        # training
        for epoch in range(epochs_CLS):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                # CLS training code here
                y_pred, loss = encoder_classifier(xb, yb)
                # Compute the loss and update the parameters
                optimizer.zero_grad()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                # Calculate the accuracy
                total_correct += (torch.argmax(y_pred, dim=1) == yb).sum().item()
                total_samples += yb.size(0)
            epoch_acc = total_correct / total_samples
            epoch_loss = total_loss / total_samples * batch_size
            train_accuracy.append(epoch_acc)
            train_loss.append(epoch_loss)
            test_accuracy.append(compute_classifier_accuracy(encoder_classifier, test_CLS_loader)/100)
            print(f'training epoch: {epoch+1} loss: {epoch_loss} training accuracy: {epoch_acc * 100}% test accuracy: {test_accuracy[-1] * 100}%')
        # testing
        print(f'testing accuracy: {compute_classifier_accuracy(encoder_classifier, test_CLS_loader)}%')
        # sanity check
        test_sentence = "That's how progress happens -- in societies and in our own lives."
        cheker = Utilities(tokenizer, encoder_classifier, "encoder")
        cheker.sanity_check(test_sentence, block_size)
        # draw a plot of training statistics
        if args.visualization:
            fig = plt.figure()
            plt.plot(train_accuracy)
            plt.plot(train_loss)
            plt.plot(test_accuracy)
            plt.title('Training accuracy, training loss, and test accuracy') 
            plt.ylabel('Accuracy or loss')
            plt.xlabel('Epoch')
            plt.legend(['Training accuracy', 'Training loss', 'Testing accuracy']) 
            plt.savefig('part1_accuracy_loss.png')
    elif args.part == 2:    
        # load training and testing datasets
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        test_LM_datasets, test_LM_loaders = [], []
        for file in ['test_LM_obama.txt', 'test_LM_wbush.txt', 'test_LM_hbush.txt']:
            inputfile = f'speechesdataset/{file}'
            with open(inputfile, 'r', encoding='utf-8') as f:
                lmtestText = f.read()
            test_LM_datasets.append(LanguageModelingDataset(tokenizer, lmtestText,  block_size))
            test_LM_loaders.append(DataLoader(test_LM_datasets[-1], batch_size=batch_size, shuffle=True))
        if args.visualization:
            # draw the histogram plot for Jaccard similarity
            jaccard_similarity = []
            train_tokens = train_LM_dataset.top_k_tokens(100)
            for i in range(3):
                # get the Jaccard similarity between classes in terms of top 100 tokens
                test_tokens = test_LM_datasets[i].top_k_tokens(100)
                set1, set2 = set(train_tokens), set(test_tokens)
                intersection = set1.intersection(set2)
                union = set1.union(set2)
                jaccard_similarity.append(round(len(intersection) / len(union), 3))
                print(f'training dataset, testing dataset {i+1}, the Jaccard similarity {jaccard_similarity[-1]}')
            labels = ['Barack Obama', 'George W. Bush', 'George H. Bush']
            fig = plt.figure()
            plt.bar(labels, jaccard_similarity) 
            plt.title('Jaccard similarity between the training and test datasets')
            plt.xlabel('Testing dataset')
            plt.ylabel('Jaccard similarity')
            plt.ylim([0, 1])
            for i in range(len(labels)):
                plt.text(labels[i], jaccard_similarity[i], str(jaccard_similarity[i]), ha='center', va='bottom')
            plt.savefig('part2_similarity.png')
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        decoder_lm = Decoder(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, device)
        print(f'number of parameters: {decoder_lm.num_params()}')
        optimizer = optim.AdamW(decoder_lm.parameters(), lr=learning_rate)
        train_perplexity, test_perplexity = [], [[] for _ in range(3)]
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            _, loss = decoder_lm(xb, yb)
            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculate the perplexity
            if (i+1) % eval_interval == 0:
                perplexity = compute_perplexity(decoder_lm, train_LM_loader, eval_iters=eval_iters)
                print(f'training iteration: {i+1}')
                print(f'Training perplexity: {perplexity}')
                train_perplexity.append(perplexity)
                for j in range(3):
                    test_perplexity[j].append(compute_perplexity(decoder_lm, test_LM_loaders[j], eval_iters=eval_iters))
                    print(f'Testing {j+1} perplexity: {test_perplexity[j][-1]}')
        # sanity check
        test_sentence = "The third source of tension is our shared interest in the rights and responsibilities of nations on nuclear weapons."
        cheker = Utilities(tokenizer, decoder_lm, "decoder")
        cheker.sanity_check(test_sentence, block_size)
        # draw the perplexity over iterations
        if args.visualization:
            fig = plt.figure()
            iterations = [100*(i+1) for i in range(5)]
            plt.plot(iterations, train_perplexity)
            for i in range(3):
                plt.plot(iterations, test_perplexity[i])
            plt.title('Training and test perplexity') 
            plt.ylabel('Perplexity')
            plt.xlabel('Iteration')
            plt.legend(['Training', 'Testing - OB', 'Testing - GWB', 'Testing - GHB']) 
            plt.savefig('part2_perplexity.png')
    elif args.part == 3: 
        # Architectural Exploration
        # load training and testing datasets
        inputfile = "speechesdataset/train_LM.txt"
        with open(inputfile, 'r', encoding='utf-8') as f:
            lmtrainText = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        test_LM_datasets, test_LM_loaders = [], []
        for file in ['test_LM_obama.txt', 'test_LM_wbush.txt', 'test_LM_hbush.txt']:
            inputfile = f'speechesdataset/{file}'
            with open(inputfile, 'r', encoding='utf-8') as f:
                lmtestText = f.read()
            test_LM_datasets.append(LanguageModelingDataset(tokenizer, lmtestText,  block_size))
            test_LM_loaders.append(DataLoader(test_LM_datasets[-1], batch_size=batch_size, shuffle=True))
        # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
        decoder_lm = Decoder_MQA(tokenizer.vocab_size, n_embd, block_size, n_layer, device)
        print(f'number of parameters: {decoder_lm.num_params()}')
        optimizer = optim.AdamW(decoder_lm.parameters(), lr=learning_rate)
        train_perplexity, test_perplexity = [], [[] for _ in range(3)]
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            # LM training code here
            _, loss = decoder_lm(xb, yb)
            # update the parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Calculate the perplexity
            if (i+1) % eval_interval == 0:
                perplexity = compute_perplexity(decoder_lm, train_LM_loader, eval_iters=eval_iters)
                print(f'training iteration: {i+1}')
                print(f'Training perplexity: {perplexity}')
                train_perplexity.append(perplexity)
                for j in range(3):
                    test_perplexity[j].append(compute_perplexity(decoder_lm, test_LM_loaders[j], eval_iters=eval_iters))
                    print(f'Testing {j+1} perplexity: {test_perplexity[j][-1]}')
        if args.visualization:
            fig = plt.figure()
            iterations = [100*(i+1) for i in range(5)]
            plt.plot(iterations, train_perplexity)
            for i in range(3):
                plt.plot(iterations, test_perplexity[i])
            plt.title('Training and test perplexity') 
            plt.ylabel('Perplexity')
            plt.xlabel('Iteration')
            plt.legend(['Training', 'Testing - OB', 'Testing - GWB', 'Testing - GHB']) 
            plt.savefig('part3_architectural_exploration_perplexity.png')
        
        # Performance Improvement
        # load data
        tokenizer = AdvancedTokenizer()
        print("New vocabulary size is", tokenizer.vocab_size)
        train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
        train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
        test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)
        if args.visualization:
            # get the Jaccard similarity between classes in terms of top 100 tokens
            train_tokens = train_CLS_dataset.top_k_tokens(100)
            test_tokens = test_CLS_dataset.top_k_tokens(100)
            for i in range(3):
                for j in range(i, 3):
                    set1, set2 = set(train_tokens[i]), set(test_tokens[j])
                    intersection = set1.intersection(set2)
                    union = set1.union(set2)
                    jaccard_similarity = len(intersection) / len(union)
                    print(f'label {i}, label {j}, the Jaccard similarity {jaccard_similarity}')
        # instantiate the encoder with classifier model
        encoder_classifier = EncoderWithClassifier(tokenizer.vocab_size, n_embd, block_size, n_head, n_layer, device, n_hidden, n_output, dropout=0.2)
        print(f'number of parameters: {encoder_classifier.num_params()}')
        encoder_classifier.train()
        # create an optimizer
        optimizer = optim.AdamW(encoder_classifier.parameters(), lr=learning_rate, weight_decay=0.2)
        # create arrays of training statistics
        train_accuracy, train_loss, test_accuracy = [], [], []
        # training
        for epoch in range(epochs_CLS):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                # CLS training code here
                y_pred, loss = encoder_classifier(xb, yb)
                # Compute the loss and update the parameters
                optimizer.zero_grad()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                # Calculate the accuracy
                total_correct += (torch.argmax(y_pred, dim=1) == yb).sum().item()
                total_samples += yb.size(0)
            epoch_acc = total_correct / total_samples
            epoch_loss = total_loss / total_samples * batch_size
            train_accuracy.append(epoch_acc)
            train_loss.append(epoch_loss)
            test_accuracy.append(compute_classifier_accuracy(encoder_classifier, test_CLS_loader)/100)
            print(f'training epoch: {epoch+1} loss: {epoch_loss} training accuracy: {epoch_acc * 100}% test accuracy: {test_accuracy[-1] * 100}%')
        # testing
        print(f'testing accuracy: {compute_classifier_accuracy(encoder_classifier, test_CLS_loader)}%')
        # # sanity check
        # test_sentence = "That's how progress happens -- in societies and in our own lives."
        # cheker = Utilities(tokenizer, encoder_classifier, "encoder")
        # cheker.sanity_check(test_sentence, block_size)
        # draw a plot of training statistics
        if args.visualization:
            fig = plt.figure()
            plt.plot(train_accuracy)
            plt.plot(train_loss)
            plt.plot(test_accuracy)
            plt.title('Training accuracy, training loss, and test accuracy') 
            plt.ylabel('Accuracy or loss')
            plt.xlabel('Epoch')
            plt.legend(['Training accuracy', 'Training loss', 'Testing accuracy']) 
            plt.savefig('part3_accuracy_loss.png')

if __name__ == "__main__":
    main()
