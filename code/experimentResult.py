import matplotlib.pyplot as plt
import re

# 로그 파일 경로 설정
log_files = ["./logs/amazon-book_seed2020_lgn_dim64_lr0.001_dec0.0001_metricrecall.txt"]  # 예시 로그 파일 리스트

# 결과 저장용 리스트 초기화
epochs = []
valid_recall20 = []
valid_ndcg20 = []
test_recall20 = []
test_ndcg20 = []

# 로그 파일 읽기 및 데이터 추출 함수
def parse_logs(log_files):
    epoch_counter = 0
    for file in log_files:
        with open(file, 'r') as f:
            for line in f:
                if line.startswith('valid'):
                    # valid 데이터 추출
                    values = list(map(float, re.findall(r"\d+\.\d+", line)))
                    valid_ndcg20.append(values[0])
                    valid_recall20.append(values[2])
                    epochs.append(epoch_counter)
                elif line.startswith('test'):
                    # test 데이터 추출
                    values = list(map(float, re.findall(r"\d+\.\d+", line)))
                    test_ndcg20.append(values[0])
                    test_recall20.append(values[2])
                    epoch_counter += 5

# 데이터 처리 실행
parse_logs(log_files)

# 그래프 그리기
plt.figure(figsize=(12, 6))

# Recall@20 및 NDCG@20 그래프 (Valid)
plt.plot(epochs, valid_recall20, label='Valid Recall@20', marker='o')
plt.plot(epochs, valid_ndcg20, label='Valid NDCG@20', marker='x')

# Recall@20 및 NDCG@20 그래프 (Test)
plt.plot(epochs, test_recall20, label='Test Recall@20', marker='s')
plt.plot(epochs, test_ndcg20, label='Test NDCG@20', marker='^')

plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Recall@20 and NDCG@20 Training Curve')
plt.legend()
plt.grid(True)

# 그래프 출력
plt.tight_layout()
plt.show()
