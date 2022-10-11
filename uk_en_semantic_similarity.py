import math
import random
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction
import datasets
import torch
from torch.utils.data import DataLoader

batch_size = 8
num_epochs = 4
model_save_path = './ftmodel'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

print('\nloading model...\n')
model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

def get_uk_data(data):
    rand_sen1 = data['sourceString']
    random.shuffle(rand_sen1)

    rand_sen2 = data['targetString']
    random.shuffle(rand_sen2)

    score1 = []
    
    for i in range(len(rand_sen1)):
        if (rand_sen1[i]==rand_sen2[i]):
            score1.append(1.0)
        else:
            score1.append(0.0)

    orig_sen1 = data['sourceString']
    orig_sen2 = data['targetString']
    score2 = [1.0] * len(orig_sen1)

    sentence1 = orig_sen1 + rand_sen1
    sentence2 = orig_sen2 + rand_sen2
    similarity_score = score2 + score1

    return sentence1, sentence2, similarity_score


def get_samples(sen1, sen2, score):
    samples = []
    
    for i in range(len(sen1)):
        samples.append(InputExample(texts=[sen1[i], sen2[i]], label=score[i]))
    
    return samples


# English Dataset

print('\nloading English dataset...\n')
en_dataset = datasets.load_dataset("stsb_multi_mt", "en")

en_scores = [x / 5.0 for x in en_dataset['test']['similarity_score']]

en_samples = get_samples(en_dataset['test']['sentence1'], en_dataset['test']['sentence2'], en_scores)


# Preparing Ukrainian Dataset

print('\nloading Ukrainian dataset...\n')
uk_dataset = datasets.load_dataset("Helsinki-NLP/tatoeba_mt", "ukr-ukr", split='test')

uk_dataset = uk_dataset.train_test_split(test_size=0.2)

uk_train = uk_dataset['train']

uk_test = uk_dataset['test']
uk_test = uk_test.train_test_split(test_size=0.5)

uk_val = uk_test['train']
uk_test = uk_test['test']

print('\npreparing Ukrainian dataset...\n')
train_sen1, train_sen2, train_score = get_uk_data(uk_train)
val_sen1, val_sen2, val_score = get_uk_data(uk_val)
test_sen1, test_sen2, test_score = get_uk_data(uk_test)

train_samples = get_samples(train_sen1, train_sen2, train_score)
val_samples = get_samples(val_sen1, val_sen2, val_score)
test_samples = get_samples(test_sen1, test_sen2, test_score)


# Initial Evaluation
uk_test_eval1 = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=batch_size, name='uk-test1', show_progress_bar=True, main_similarity=SimilarityFunction.COSINE)
print('\nevaluating (Ukrainian)...\n')
uk_eval1 = uk_test_eval1(model, output_path=model_save_path)

en_test_eval1 = EmbeddingSimilarityEvaluator.from_input_examples(en_samples, batch_size=batch_size, name='en-test1', show_progress_bar=True, main_similarity=SimilarityFunction.COSINE)
print('\nevaluating (English)...\n')
en_eval1 = en_test_eval1(model, output_path=model_save_path)


# Fine Tune on Ukrainian

train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=batch_size)

train_loss = losses.CosineSimilarityLoss(model=model)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_samples, name='uk-val', main_similarity=SimilarityFunction.COSINE)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1) #10% of train data for warm-up

print('\nfine-tuning model on Ukrainian dataset...\n')


model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          show_progress_bar = True
          )


model = SentenceTransformer(model_save_path)

# Final Evaluation

print('\nevaluating (Ukrainian)...\n')
uk_test_eval2 = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=batch_size, name='uk-test2', show_progress_bar=True, main_similarity=SimilarityFunction.COSINE)
uk_eval2 = uk_test_eval2(model, output_path=model_save_path)

print('\nevaluating (English)...\n')
en_test_eval2 = EmbeddingSimilarityEvaluator.from_input_examples(en_samples, batch_size=batch_size, name='en-test2', show_progress_bar=True, main_similarity=SimilarityFunction.COSINE)
en_eval2 = en_test_eval2(model, output_path=model_save_path)


# Results

print('Initial Ukrainian Evaluation: ', uk_eval1)
print('Final Ukrainian Evaluation: ', uk_eval2)

print('Initial English Evaluation: ', en_eval1)
print('Final English Evaluation: ', en_eval2)


