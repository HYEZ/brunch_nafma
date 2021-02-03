#-*- encoding: utf-8 -*-
# inference using statistics, doc2vec
# mf removed after evaluation
# word2vec removed due to gensim license
# doc2vec removed and changed from gensim to tensorflow

import os, sys 
sys.path.append('..')
sys.path.append('../../')

import pdb 
import tqdm
import json
import datetime
from util import iterate_data_files
import glob
from itertools import chain
import numpy as np
import pandas as pd
from scipy import spatial

import config as conf

np.random.seed(1)


# recommend a popular sequential articles thas was read in the dup dates
# 연속 조회 통계 기반으로 함께 조회된 글을 예측하는 코드
def find_dup_seq(viewer): # viewer: 독자의 id
  # 구독작가 여부에 다라 예측 결과를 2그룹으로 분리함, 구독 중인 작가의 글을 예측 리스트의 앞부분, 구독 중이 아닌 작가의 글을 뒷부분에 넣음
  recommends1 = [] 
  recommends2 = [] 
  if viewer in t_followings:
    followings = t_followings[viewer] # 구독 작가 목록
  else:
    followings = []

  if viewer in t_reads_dup: # 예측 대상 사용자들이 최근에 조회한 글 목록 (중첩기간, 없으면 최근에 조회한 10개)
    reads_org = t_reads_dup[viewer]
    reads = sorted(set(reads_org), key=lambda x: reads_org.index(x))  # dedup and keep order
    reads.reverse()   # the later the better (최신글이 앞에)
    num_reads = len(reads) # 사용자가 최근에 조회한 글의 개수
    for read in reads:
      if read in seq_read:
        seqs = seq_read[read]
        for seq in seqs:
          if seq not in t_reads[viewer]: # 독자가 읽은 글이 아닌 거만 추천하려고
            if (seq not in recommends1) and (seq not in recommends2): # 조회 예측 결과에 들어 있지 않은 글
              writer = seq.split("_")[0]
              if writer in followings: # 구독하는 작가의 글이라면 
                recommends1.append(seq)
              else: # 구독하는 작가의 글이 아니라면
                recommends2.append(seq)
              break
          #if num_reads > 100: break
          if num_reads > 50: break  # 엔트로피 성능을 높이기 위한 변경 (최근 조회가 50건이 넘으면 seqs에 들어있는 글 중에서 가장 빈도가 높은 1개의 글만 예측에 사용함 => 예측 사용자당 100건 의 글을 예측할때 CF로 너무 많은 글을 추천하지 않게 하여 엔트로피 성능을 위한 추천을 추가할 여분을 만들기 위함)

      if num_reads < 50: # 최근에 조회한 글의 개수가 50보다 적으면 => 최근에 조회한 글 바로 이전에 조회된 글을 이용한 예측 추가
        if read in prev_read:
          seqs = prev_read[read]
          for seq in seqs:
            if seq not in t_reads[viewer]:
              if (seq not in recommends1) and (seq not in recommends2):
                writer = seq.split("_")[0]
                if writer in followings:
                  recommends1.append(seq)
                else:
                  recommends2.append(seq)
                break
  return recommends1, recommends2


# add new articles of following writer in reverse order (recent first)
# need to experiment to change windows and order for the case over 100
# 선호 작가의 새 글을 추천
def find_new_articles(viewer):
  recommends1 = [] # 중첩 기간에 사용자가 글을 읽은 선호 작가의 최신 글 저장
  recommends2 = [] # 중첩 기간에 사용자가 글을 읽지 않은 선호 작가의 최신 글 저장

  dup_read_writers = {}
  if viewer in t_reads_dup: # t_reads_dup: 예측 대상 사용자가 중첩 기간에 읽은 글을 담음
    reads = t_reads_dup[viewer]
    for read in reads:
      writer = read.split("_")[0] # 그 글의 작가
      if writer not in dup_read_writers:
        dup_read_writers[writer] = 1  # 예측 대상 사용자가 중첩 기간에 읽은 글의 작가 목록을 담음

  read_writers = {} # 모든 기간동안 읽은 글의 작가 목록을 저장함
  if viewer in t_reads:
    reads = t_reads[viewer]
    for read in reads:
      writer = read.split("_")[0]
      if writer not in read_writers:
        read_writers[writer] = 1

  if viewer in t_followings:
    followings = t_followings[viewer] # 독자들의 구독 작가 목록 저장
  else:
    followings = []
  if viewer in t_non_follow:
    non_follow = t_non_follow[viewer] # 구독하지는 않지만 글을 많이 조회한 작가 목록 저장
  else:
    non_follow = [] # 예측 대상 사용자가 구독하는 작가가 하나도 없고, 조회한 글도 하나도 없는 경우 콘텐츠 기반으로 예측할 수 없으므로 빈 리스트 리턴하며 콘텐츠 기반 필터링 종료함

  if len(followings) == 0:
    if len(non_follow) == 0:
      #print("no followings no freq for", viewer)
      return recommends1, recommends2

  # 조회 수 많은 순서로 정렬
  # 앞쪽에 예측한 글의 정답 여부가 뒤쪽에 예측한 글의 정답 여부보다 높은 성능 점수로 계산됨 => 구독 작가 중에서 조회가 많은 구독 작가의 새 글을 예측 목록 앞에 넣음
  followings_sorted_stats = []  # 가장 높은 우선순위
  if viewer in t_reads:
    followings_cnt = {} # 작가별 조회 수 저장
    all_reads = t_reads[viewer] # 해당 독자가 읽은 모든 글
    for read in all_reads:
      writer = read.split("_")[0]
      if writer in followings:
        if writer in followings_cnt:
          followings_cnt[writer] += 1
        else:
          followings_cnt[writer] = 1
    # 조회 수 순서대로 정렬
    followings_cnt_sorted = sorted(followings_cnt.items(), key=lambda kv: kv[1], reverse=True)

    for writer, cnt in followings_cnt_sorted:
      if writer in followings:
        followings_sorted_stats.append(writer) # 작가 아이디만 가져옴 (조회수가 많은 순서로 정렬한 구독 작가 리스트)

  # sort by d2v
  # 키워드가 비슷한 작가의 글을 추천
  followings_sorted = []  # 2nd priority, 주어진 예측 대상 사용자의 구독 작가들을 선호도 순서로 저장
  if viewer in sentences_df_indexed.index: # 인덱스는 독자와 작가의 아이디
    followings_sim = []
    sims = {}
    for writer in followings: # 독자의 구독 작가들
      if writer in sentences_df_indexed.index:
        sim = similarity(viewer, writer) # 독자와 작가의 코사인 유사도
        if sim not in sims:
          sims[sim] = 1
        else:
          sim -= 0.000001 # 동일한 유사도가 존재하면 유사도 조금 낮춤
          if sim not in sims:
            sims[sim] = 1
          else:
            sim -= 0.000001
            if sim not in sims:
              sims[sim] = 1
        
        followings_sim.append([writer, sim]) # 작가와 독자의 유사도 저장
        
    followings_sim_sorted = sorted(followings_sim, key=lambda x:x[1], reverse=True) # 유사도 순으로 작가 정렬
    
    for item in followings_sim_sorted:
      followings_sorted.append(item[0])

    for writer in followings_sorted:
      if writer not in followings_sorted_stats:
        followings_sorted_stats.append(writer) # 유사도가 높은 순서로 정렬한 구독 작가 추가

  for writer in followings:
    if writer not in followings_sorted_stats: # 조회수와 유사도로 정렬한 구독작가에 포함하지않은 구독작가를 넣어줌
      followings_sorted_stats.append(writer)

  followings = followings_sorted_stats

  if len(non_follow) > 0:
    if len(followings) < 10:
        # 구독 작가가 10 미만이면 글을 많이 읽은 작가를 추가해줌
      followings += non_follow[:2]  # for ent

  if viewer in t_reads:
    reads = t_reads[viewer] # 이전에 읽은 글은 추천에서 제외하기 위해 이전에 읽은 글들 가져옴
  else:
    reads = []
    #print("no previous reads for", viewer)

  for writer in followings: # 추천할 작가 리스트가 저장됨
    if writer not in writer_articles: continue
    articles = writer_articles[writer] 
    articles_sorted = sorted(articles, key=lambda x: x[1], reverse=False) # 글을 오래된 순서대로 읽어서 articles_sorted에 저장
    for article, reg_datetime in articles_sorted:
      if reg_datetime <= "20190221000000": continue  # 기간을 제한해서 엔트로피 성능 높임
      if reg_datetime >= "20190315000000": break
      if article in reads: # 이미 읽은 글은 추천안함
        #print("found article already read")
        continue
      if article not in recommends1 and article not in recommends2: # 중복 추천 안하게
        if writer in dup_read_writers:
          recommends1.append(article)
        #else:
        elif writer != "@brunch" or writer in read_writers: # 엔트로피 성능 높임
          recommends2.append(article)

  # order should be changed for ndcg reason
  # recommend1이 70이 넘는 경우 cbf 추천이 너무 많아지므로 190301 ~ 190313까지 등록된 글로 cbf 추천 제한함
  if len(recommends1) > 70:
    recommends1 = []
    recommends2 = []
    for writer in followings:
      if writer not in writer_articles: continue
      articles = writer_articles[writer]
      articles_sorted = sorted(articles, key=lambda x: x[1], reverse=False)
      for article, reg_datetime in articles_sorted:
        # 글 등록 날짜만 달라지고 위와 동일
        if reg_datetime <= "20190301000000": continue  # smaller window will make higher ent
        if reg_datetime >= "20190313000000": break
        if article in reads:
          #print("found article already read")
          continue
        if article not in recommends1 and article not in recommends2:
          if writer in dup_read_writers:
            recommends1.append(article)
          #else:
          elif writer != "@brunch" or writer in read_writers: # for higher ent
            recommends2.append(article)

  return recommends1, recommends2


def read_test_user():
  print("read test user set", user_file)
  with open(user_file, "r") as fp:
    for line in fp:
      viewer_id = line.strip()
      t_users[viewer_id] = 1


def read_followings():
  print("read viewer followings")
  with open(conf.data_root + "users.json", "r") as fp:
    for line in fp:
      viewer = json.loads(line)
      if viewer['id'] in t_users:
        t_followings[viewer['id']] = viewer['following_list']
        if len(viewer['keyword_list']) > 0:
          t_keywords[viewer['id']] = []
          for keyword in viewer['keyword_list']:
            t_keywords[viewer['id']].append(keyword['keyword'])


# may need to write pickle for this
# 모든 글 조회 데이터를 읽어 연속 조회 통계를 seq_reads에 저장. 예측 대상 사용자들의 글 조회에 대해서도 예측을 위해 t_reads_dup에 저장.
def read_reads():
  print("read reads of all users")
  files = sorted([path for path, _ in iterate_data_files('2018100100', '2019030100')])
  for path in tqdm.tqdm(files, mininterval=1):
    date = path[11:19]
    for line in open(path):
      tokens = line.strip().split()
      user_id = tokens[0]
      reads = tokens[1:]
      if len(reads) < 1: continue
      if user_id in t_users: # 예측 대상 사용자들
        if user_id in t_reads: # 예측 대상 사용자들의 조회인 경우 t_reads에 저장
          t_reads[user_id] += reads 
          # 동일한 사용자가 서로 다른 시간에 접속하여 조회한 경우 조회한 글을 t_reads의 하나의 키에 시간의 순서를 유지하면서 모아서 저장
        else:
          t_reads[user_id] = reads
        
        # 테스트 데이터랑 중첩된 날짜에 대해선 t_reads_dup에 저장
        if date >= "20190222":
          if user_id in t_reads_dup:
            t_reads_dup[user_id] += reads
          else:
            t_reads_dup[user_id] = reads

        reads_set = set(reads) # 같은 세션 내에서 여러번 조회한 경우 중복을 제거하기 위함
        for read in reads_set:
          writer = read.split("_")[0]
          # t_followings: 각 유저의 구독 작가 목록 딕셔너리
          if (user_id not in t_followings) or (writer not in t_followings[user_id]):
            if user_id in t_non_follows:
              if writer in t_non_follows[user_id]: # 구독 관계가 없는 작가에 대해 읽은 횟수를 저장 (키=독자아이디/작가아이디, 값=읽은 횟수)
                t_non_follows[user_id][writer] += 1 
              else:
                t_non_follows[user_id][writer] = 1
            else:
              t_non_follows[user_id] = {}
              t_non_follows[user_id][writer] = 1

      num_reads_n1 = len(reads)-1
    
      # 바로 다음 글 조회 기록 저장
      for i, read in enumerate(reads):
        if i < num_reads_n1:
          if read == reads[i+1]: continue   # when two continous reads are the same (바로 다음글이 같은 글이면 continue)
          if read in seq_reads:
            if reads[i+1] in seq_reads[read]:
              seq_reads[read][reads[i+1]] += 1 # seq_reads: 연속 조회 통계 저장 (바로 다음 조회한 글의 아이디를 저장!!)
            else:
              seq_reads[read][reads[i+1]] = 1
          else:
            seq_reads[read] = {} 
            seq_reads[read][reads[i+1]] = 1
    
      # 바로 이전 글 조회 기록 저장
      for i, read in enumerate(reads):
        if i < num_reads_n1:
          nread = reads[i+1]
          if read == nread: continue   # when two continous reads are the same
          if nread in prev_reads:
            if read in prev_reads[nread]:
              prev_reads[nread][read] += 1
            else:
              prev_reads[nread][read] = 1
          else:
            prev_reads[nread] = {}
            prev_reads[nread][read] = 1
    
  # 예측 대상 사용자들이 중첩 기간에 조회한 글이 하나도 없는 경우, 가장 최근에 읽은 글 10개를 t_reads_dup에 저장
  for user in t_reads:
    if user not in t_reads_dup:
      t_reads_dup[user] = t_reads[user][-10:] # t_reads_dup: 예측 대상 사용자들의 글 조회


def determine_non_follow():
  print("find not following but favorite writers")
  for user in t_non_follows:
    writers = t_non_follows[user]
    writers_sorted = sorted(writers.items(), key=lambda x: x[1], reverse=True)
    if len(writers_sorted) < 3: tops = len(writers_sorted)
    else: tops = 3
    if writers_sorted[0][1] < 5: continue
    t_non_follow[user] = []
    for i in range(tops):
      if writers_sorted[i][1] < 5: break
      t_non_follow[user].append(writers_sorted[i][0])


# may need to write pickle for this
# 연속 조회 통계에서 가장 많이 연속 조회되는 3개의 글을 찾아 seq_read에 저장
def determine_seq_read(): 
  print("find co-occurence of articles")
  for article in seq_reads:
    reads = seq_reads[article]
    # 바로 다음 글을 읽은 횟수인 seq_reads 에 대해 연속 조회가 많은 순서대로 sort함
    reads_sorted = sorted(reads.items(), key=lambda kv:kv[1], reverse=True)
    if len(reads_sorted) < 3: tops = len(reads_sorted)
    else: tops = 3 
    seq_read[article] = []
    for i in range(tops):
      if reads_sorted[i][1] < 2: break # 연속 조회수가 2개 미만인 경우 저장하지 않음
      seq_read[article].append(reads_sorted[i][0])  # 바로 다음 조회한 글 아이디들, 최대 3개의 리스트
   
  # 이전 글에 대해서도 똑같이 수행
  for article in prev_reads:
    reads = prev_reads[article]
    reads_sorted = sorted(reads.items(), key=lambda kv:kv[1], reverse=True)
    if len(reads_sorted) < 3: tops = len(reads_sorted)
    else: tops = 3
    prev_read[article] = []
    for i in range(tops):
      if reads_sorted[i][1] < 2: break
      prev_read[article].append(reads_sorted[i][0])


# may need to write pickle for this
# prepare article info
def read_article_meta():
  print("build article id and registration time for each writer")
  with open(conf.data_root + "metadata.json", "r") as fp:
    for line in fp:
      article = json.loads(line)
      article_id = article['id']
      writer_id = article['user_id']
      reg_datetime = datetime.datetime.fromtimestamp(article['reg_ts']/1000).strftime("%Y%m%d%H%M%S")
      if writer_id in writer_articles:
        writer_articles[writer_id].append([article_id, reg_datetime])
      else:
        writer_articles[writer_id] = [[article_id, reg_datetime]]


# 조회기록이 없거나 조회가 있더라도 부족한 예외적인 상황을 위해
# 예외상황에 추천할 글을 준비하는 함수
# dedup_recs are in reverse order (the sooner the better)
def prepare_dedup_recs():
  print("prepare recommendations with old read or no read")
  dedup_recs = []
  for writer in writer_articles:
    articles = writer_articles[writer] # writer_articles: 작가별로 작성한 글과 등록일시를 가진 딕셔너리
    if len(articles) < 2: continue
    # 글을 2개이상 읽은 경우 모든 글과 등록일시를 dedup_recs에 저장
    for item in articles:
      dedup_recs.append(item)

  dedup_recs_sorted = sorted(dedup_recs, key=lambda x: x[1], reverse=True) # 최신순으로 정렬
  dedup_recs = []
  for article, reg_datetime in dedup_recs_sorted:
    dedup_recs.append(article) # 글만 저장

  return dedup_recs


# 예외 상황에 추천할 글을 결정하는 함수
# cf + cbf 추천 합이 100이 넘지 않는 경우 => 엔트로피 높이기
# 모든 사용자를 기준으로 겹치지 않게 최신성이 있는 글 추천하기
def add_dedup_recs(viewer, rec100, dedup_recs):
  rec100_org = rec100.copy() # cf + cbf 로 추천된 결과 리스트
  if viewer in t_reads:
    reads = t_reads[viewer]
    writers = {}
    for read in reads:
      writer = read.split("_")[0]
      if writer not in writers:
        writers[writer] = 1

    i = 0
    while i < len(dedup_recs):
      writer = dedup_recs[i].split("_")[0]
      if (dedup_recs[i] not in all_recs) and (dedup_recs[i] not in rec100) and (writer in writers):
        rec100.append(dedup_recs[i])
      i += 1
      if len(rec100) >= 100:
        break
  i = 0
  while i < len(dedup_recs):
    if len(rec100) >= 100:
      break
    if (dedup_recs[i] not in all_recs) and (dedup_recs[i] not in rec100):
      rec100.append(dedup_recs[i])
    i += 1
    if len(rec100) >= 100:
      break
  return rec100


def add_dedup_recs_d2v(viewer, rec100, dedup_recs):
  top_writers = {}
  if viewer in model.docvecs:
    tops = model.docvecs.most_similar(viewer, topn=200)
    for top in tops:
      top_writers[top[0]] = 1

  if len(top_writers) > 0:
    i = 0
    recs = []
    while i < len(dedup_recs):
      rec = dedup_recs[i]
      rec_writer = rec.split("_")[0]
      if rec_writer in top_writers:
        if (rec not in all_recs) and (rec not in rec100):
          rec100.append(rec)
      i += 1
      if len(rec100) >= 100:
        break

  if len(rec100) < 100:
    i = 0
    while i < len(dedup_recs):
      if (dedup_recs[i] not in all_recs) and (dedup_recs[i] not in rec100):
        rec100.append(dedup_recs[i])
      i += 1
      if len(rec100) >= 100:
        break

  return rec100


def most_similar(user_id, size):
    user_index = sentences_df_indexed.loc[user_id]['index']
    dist = final_doc_embeddings.dot(final_doc_embeddings[user_index][:,None])
    closest_doc = np.argsort(dist,axis=0)[-size:][::-1]
    furthest_doc = np.argsort(dist,axis=0)[0][::-1]

    result = []
    for idx, item in enumerate(closest_doc):
        user = sentences[closest_doc[idx][0]].split()[0]
        dist_value = dist[item][0][0]
        result.append([user, dist_value])
    return result


def similar(user_id, writer_id):
    user_index = sentences_df_indexed.loc[user_id]['index']
    writer_index = sentences_df_indexed.loc[writer_id]['index']
    dist = final_doc_embeddings[user_index].dot(final_doc_embeddings[writer_index])
    #print('{} - {} : {}'.format(user_id, writer_id, dist))
    return dist

def similarity(user_id, writer_id):
    if user_id in sentences_df_indexed.index and writer_id in sentences_df_indexed.index:
        user_index = sentences_df_indexed.loc[user_id]['index']
        writer_index = sentences_df_indexed.loc[writer_id]['index']
        sim = spatial.distance.cosine(final_doc_embeddings[user_index], final_doc_embeddings[writer_index])
        #print('{} - {} : {}'.format(user_id, writer_id, sim))
        return sim

if __name__ == "__main__":
  if len(sys.argv) < 2:
    user_file = conf.data_root + "predict/test.users"
  elif sys.argv[1] == "test":
    user_file = conf.data_root + "predict/test.users"
  else:
    user_file = conf.data_root + "predict/dev.users"

  print("load d2v model")
  #files = glob.glob('./res/writer_user_doc.txt')
  files = glob.glob('./res/writer_user_sentences_keyword.txt')
  words = []
  for f in files:
      file = open(f)
      words.append(file.read())
      file.close()

  words = list(chain.from_iterable(words))
  words = ''.join(words)[:-1]
  sentences = words.split('\n')
  sentences_df = pd.DataFrame(sentences)

  sentences_df['user'] = sentences_df[0].apply(lambda x : x.split()[0])
  sentences_df['words'] = sentences_df[0].apply(lambda x : ' '.join(x.split()[1:]))
  sentences_df_indexed = sentences_df.reset_index().set_index('user')

  #final_doc_embeddings = np.load('./doc_embeddings.npy')
  final_doc_embeddings = np.load('./doc_embeddings_keyword.npy')

  t_users = {}   # all test_users # 예측 대상 사용자들을 key로함 
  t_keywords = {} # keywords # 검색 키워드
  t_followings = {}   # following writer list for test users # 예측 대상 사용자들의 구독 작가
  t_non_follows = {}   # non-follow but many reads writer list for test users # 많이 조회한 작가
  t_non_follow = {}   # top3 non-follow but many reads writer list for test users # 최대 3명의 많이 조회한 작가
  t_reads = {}        # read articles for test users # 예측 대상 사용자들이 조회한 전체 글
  t_reads_dup = {}    # read articles during dup dates for test users (2/22~) # 예측 대상 사용자들이 중첩 기간에 조회한 글
  writer_articles = {} # 작가들의 아이디를 key, 작가가 작성한 글 id가 value
  seq_reads = {}      # sequentially read articles # 순방향 연속 조회 전체 통계
  seq_read = {}       # top3 sequentially read articles # 순방향 연속 최대 조회글 최대 3건
  prev_reads = {}      # sequentially read articles # 역방향 연속 조회 전체 통계
  prev_read = {}       # top3 sequentially read articles # 역방향 연속 최대 조회글 최대 3건
  all_recs = {} # 모든 사용자의 모든 추천글을 Key로 저장함

  read_test_user() # 예측대상 사용자의 아이디 데이터를 읽음

  read_followings() # 구독 데이터 읽음

  read_reads() # 글의 메타데이터를 읽음
 
  determine_seq_read() # 연속 조회 통계와 관련된 딕셔너리 변수 값 설정
  determine_non_follow() # 예측대상 사용자들이 많이 조회한 작가를 저장하는 딕셔너리 값 설정

  read_article_meta() # 작가별로 작성한 글을 저장하는 딕셔너리 변수 값 설정
  dedup_recs = prepare_dedup_recs() # cf , cbf로 추천하는 글 이외의 추가로 중복없이 신규 글 우선으로 추천
  
                             
  ##### -----> 앙상블 구현 <----- #####
  of1 = open(conf.root + "/submission/recommend_1.txt", "w") # cf
  of2 = open(conf.root + "/submission/recommend_2.txt", "w") # cbf
  of12 = open(conf.root + "/submission/recommend.txt", "w") # 앙상블
  print("start recommending articles")
  num_recommended = 0 # 예측 대상 사용자 총합
  num_recommended1and2 = 0 # cf, cbf로 모두 추천글을 생성한 사용자 총합
  num_recommended1 = 0 # cf로만 추천한 사용자 총합 
  num_recommended2 = 0 # cbf로만 추천한 사용자 총합
  num_recommends1 = 0 # cf 로 추천한 글 개수 총합
  num_recommends2 = 0 # cbf로 추천한 글 개수 총합
  num_recommends1or2 = 0 # cf, CBF로 추천한 글 개수 총합
                             
  for cnt, viewer in enumerate(t_users):
    if (cnt % 100) == 99: print(str(cnt+1), "/", str(len(t_users)))
    
    # 1. CF 실행
    recommends11, recommends12 = find_dup_seq(viewer)
    recommends1 = recommends11 + recommends12
    if len(recommends1) > 0:
      of1.write(viewer + " " + " ".join(recommends1[:100]) + "\n") # 최대 100개만 저장
    num_recommend1 = len(recommends1[:100]) 
    num_recommends1 += num_recommend1
                            
    # 2. CBF 실행
    recommends21, recommends22 = find_new_articles(viewer)
    recommends2 = recommends21 + recommends22

    if len(recommends2) > 0:
      of2.write(viewer + " " + " ".join(recommends2[:100]) + "\n") # 최대 100개만 저장
    num_recommend2 = len(recommends2[:100])
    num_recommends2 += num_recommend2
                             
   
    
    if num_recommend1 > 0: # cf, cbf 둘다 추천 결과가 있으면
      if num_recommend2 > 0:
        num_recommended1and2 += 1
      else: # cf만 있으면
        num_recommended1 += 1
    elif num_recommend2 > 0: # cbf만 있으면
      num_recommended2 += 1
    
    recommends_1or2 = recommends11.copy() # 최종 추천글을 담을 리스트. cf중에 우선순위 1번째 가져옴 (예측 대상 사용자들이 최근에 조회한 글들에 포함된 구독작가의 새글)

    for rec in recommends21: # 두번째 우선순위인 cbf로 추천한 (구독작가 또는 선호작가의 새글)
      if rec not in recommends_1or2: # 중복추천 방지
        recommends_1or2.append(rec)

    for rec in recommends12: # 3번째 우선순위 (예측 대상 사용자들이 최근에 조회한 글들에 포함된 구독작가가 아닌 작가의 글 중 새글)
      if rec not in recommends_1or2: # 중복 추천 방지
        recommends_1or2.append(rec)

    for rec in recommends22: # 4번재 우선순위 (doc2vec 유사도기반 키워드가 비슷한 작가의 새글)
      if rec not in recommends_1or2: # 중복추천 방지
        recommends_1or2.append(rec)

    num_recommends1or2 += len(recommends_1or2[:100]) # 100개로 자름
    num_recommended += 1

    if len(recommends_1or2[:100]) < 100: # 100보다 작으면 예외상황으로 추천
      #recommends_1or2 = add_dedup_recs_d2v(viewer, recommends_1or2, dedup_recs)
      recommends_1or2 = add_dedup_recs(viewer, recommends_1or2, dedup_recs)

    if len(recommends_1or2) < 100:
      pdb.set_trace()

    for rec in recommends_1or2[:100]:
      if rec not in all_recs:
        all_recs[rec] = 1

    of12.write(viewer + " " + " ".join(recommends_1or2[:100]) + "\n") # 총 100개 저장

  of1.close()
  of12.close()
  of2.close()

  print(num_recommended, num_recommended1and2, num_recommended1, num_recommended2)
  print(num_recommends1, num_recommends1or2-num_recommends1)
