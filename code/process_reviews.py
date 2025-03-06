import argparse
import time
import datetime  # ⏳ 시간 형식 변환을 위한 모듈 추가
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# BERT 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# BERT 임베딩 변환 함수 (CLS 토큰 사용)
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# 리뷰 임베딩 변환 및 저장 함수
def process_reviews(input_file, output_file):
    start_time = time.time()  # ⏳ 시작 시간 기록
    
    user_item_embeds = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"🔄 Processing {input_file} ({len(lines)} reviews)...")
    
    for line in tqdm(lines, desc="Processing Reviews"):
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        
        user, item, review_text = parts
        user, item = int(user), int(item)
        embed = get_bert_embedding(review_text)  # BERT 임베딩 변환
        user_item_embeds[(user, item)] = embed

    # numpy 파일로 저장
    np.save(output_file, user_item_embeds)
    print(f"✅ {output_file} 저장 완료! ({len(user_item_embeds)} embeddings)")

    # 실행 시간 계산 ⏳
    end_time = time.time()
    elapsed_time = end_time - start_time  # 초 단위 실행 시간

    # ⏳ hh:mm:ss 형식으로 변환
    elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))

    print(f"⏳ Total Processing Time: {elapsed_str}")
    
# ✅ 명령줄 인자 받기 (argparse 사용)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process reviews and generate BERT embeddings")
    parser.add_argument("--input", type=str, required=True, help="Path to the input review file")
    parser.add_argument("--output", type=str, required=True, help="Path to the output .npy file")

    args = parser.parse_args()
    
    # 입력받은 경로로 process_reviews 실행
    process_reviews(args.input, args.output)
