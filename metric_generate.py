import json
import numpy as np
import math

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import rouge_n
from distinct_n import distinct_n_sentence_level



def add_score(score_term):
    score_average = 0
    for i in score_term:
        score_average += i
    return score_average/len(score_term)

def std(score_term):
    score_average = 0
    for i in score_term:
        score_average += i
    return score_average/len(score_term)

def moving_average(interval, windowsize):
    window = np.ones(int(windowsize)) / float(windowsize)
    re = np.convolve(interval, window, 'same')
    return re

with open('a-act_cvae_60epoch.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
cat = ['inform', 'question', 'directive', 'commissive']
with open('../data/raw/act/act_testset_classify.txt', 'r', encoding='utf-8') as f:
    lines2 = f.readlines()

post, response, real_label, clf_result, inform, question, directive, commissive ,no_label, true_clf_decoder = [], [], [], [], [], [], [], [], [], []

for id, line in enumerate(lines):
    line = json.loads(line)
    line2 = json.loads(lines2[id])
    post.append(line['post'])
    response.append(line['response'])
    real_label.append(int(line2['post_label_act'][0]))
    clf_result.append(line['clf_result'])
    question.append(line['question'])
    inform.append(line['inform'])
    directive.append(line['directive'])
    commissive.append(line['commissive'])
    no_label.append(line['no_label'])
    if int(line2['post_label_act'][0]) == 0:
        true_clf_decoder.append(line['inform'])
    elif int(line2['post_label_act'][0]) == 1:
        true_clf_decoder.append(line['question'])
    elif int(line2['post_label_act'][0]) == 2:
        true_clf_decoder.append(line['directive'])
    else:
        true_clf_decoder.append(line['commissive'])


acc = 0
for i in range(len(real_label)):
    if real_label[i] == clf_result[i]:
        acc += 1

ref = response
# for name in ['inform', 'question', 'directive', 'commissive', 'pretrain', 'slect by category']:
for name in ['slect by category']:
    if name == 'inform':
        res = inform
    elif name == 'question':
        res = question
    elif name == 'directive':
        res = directive
    elif name == 'commissive':
        res = commissive
    elif name == 'pretrain':
        res = no_label
    else:
        res = true_clf_decoder

    bleus_score = []
    meteors_score = []
    rouges_score = []
    refs = []
    hyps = []
    for i in range(0, len(ref)):
        refs.append([ref[i]])
        hyps+= res[i]
        if len(' '.join(res[i]).lower()) > 1:
            bleus_score.append(sentence_bleu([ref[i]], res[i], weights=(1, 0, 0, 0)) * 100)
            meteors_score.append(round(meteor_score([' '.join(ref[i]).lower()], ' '.join(res[i]).lower()), 4) * 100)
            rouge_2, _, _ = rouge_n(' '.join(ref[i]).lower(), ' '.join(res[i]).lower())
            rouges_score.append(rouge_2 * 100)
    mean_bleu = add_score(bleus_score)
    mean_meteor = add_score(meteors_score)
    mean_rouge = add_score(rouges_score)
    SEM_bleu = np.sqrt(np.sum(np.power(np.array(bleus_score) - mean_bleu, 2)) / len(bleus_score)) / math.sqrt(
        len(bleus_score))
    SEM_meteor = np.sqrt(np.sum(np.power(np.array(meteors_score) - mean_meteor, 2)) / len(meteors_score)) / math.sqrt(
        len(meteors_score))
    SEM_rouge = np.sqrt(np.sum(np.power(np.array(rouges_score) - mean_rouge, 2)) / len(rouges_score)) / math.sqrt(
        len(rouges_score))
    print('*******{}**********'.format(name))
    print('BLEU: ' + str(mean_bleu) + 'error bar:' + str(SEM_bleu))
    print('METEOR: ' + str(mean_meteor) + 'error bar:' + str(SEM_meteor))
    print('ROUGE: ' + str(mean_rouge) + 'error bar:' + str(SEM_rouge))
    print('Distinct-2: ' + str(distinct_n_sentence_level(hyps, 2)))

print('***********ACC***********')
print(acc/len(real_label))

