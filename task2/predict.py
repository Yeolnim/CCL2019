import jieba
import numpy as np
import pandas as pd
from keras.models import load_model

model = load_model('task2.h5')

test = pd.read_csv('task2_test.csv')
test['sen_cut'] = test['joke'].apply(jieba.lcut)

X_test = test['sen_cut'].apply(lambda x: ' '.join(x)).tolist()
# y_test = pd.get_dummies((np.asarray(test["label"])))
text = np.array(X_test)

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import time

vocab_size = 50000
maxlen = 70

print("开始统计语料的词频信息...")
t = Tokenizer(vocab_size)
t.fit_on_texts(text)
word_index = t.word_index
print('完整的字典大小：', len(word_index))

print("开始序列化句子...")
X_test = t.texts_to_sequences(X_test)
print("开始对齐句子序列...")
X_test = pad_sequences(X_test, maxlen=maxlen, padding='post')
print("完成！")

import copy

small_word_index = copy.deepcopy(word_index) # 防止原来的字典被改变
x = list(t.word_counts.items())
s = sorted(x, key=lambda p:p[1], reverse=True)
print("移除word_index字典中的低频词...")
for item in s[20000:]:
    small_word_index.pop(item[0]) # 对字典pop
print("完成！")
print(len(small_word_index))
print(len(word_index))

print(type(X_test))
# print(type(y_test))

print(X_test.shape)

print(test['joke'][:2])

predicted = np.array(model.predict(X_test))
test_predicted=np.argmax(predicted,axis=1)

print(len(test_predicted))
print(test_predicted)


result=pd.read_csv('task2_test.csv',encoding='utf-8')
result['label']=test_predicted
result.to_csv('result.csv', encoding='utf-8',index=False)
