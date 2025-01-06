import matplotlib.pyplot as plt
import re

# 로그 파일 경로 설정
recall_log_file = "./logs/amazon-book_seed2020_lgn_dim64_lr0.001_dec0.0001_metricrecall.txt"
ndcg_log_file = "./logs/amazon-book_seed2020_lgn_dim64_lr0.001_dec0.0001_metricndcg.txt"

# 결과 저장용 리스트 초기화
valid_recall20 = []
valid_ndcg20 = []
test_recall20 = []
test_ndcg20 = []

# 로그 파일 읽기 및 데이터 추출 함수
def parse_logs_recall(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('valid'):
                # valid 데이터 추출
                values = list(map(float, re.findall(r"\d+\.\d+", line)))
                valid_recall20.append(values[2])
            elif line.startswith('test'):
                # test 데이터 추출
                values = list(map(float, re.findall(r"\d+\.\d+", line)))
                test_recall20.append(values[2])

def parse_logs_ndcg(log_file):
    with open(log_file, 'r') as f:
        for line in f:
            if line.startswith('valid'):
                # valid 데이터 추출
                values = list(map(float, re.findall(r"\d+\.\d+", line)))
                valid_ndcg20.append(values[0])
            elif line.startswith('test'):
                # test 데이터 추출
                values = list(map(float, re.findall(r"\d+\.\d+", line)))
                test_ndcg20.append(values[0])

# 데이터 처리 실행
parse_logs_recall(recall_log_file)
parse_logs_ndcg(ndcg_log_file)

# 최대 길이 에포크 설정
epochs = [i * 5 for i in range(max(len(valid_recall20), len(valid_ndcg20)))]

# 그래프 그리기
plt.figure(figsize=(12, 6))

# Recall@20 및 NDCG@20 그래프 (Valid)
plt.plot(epochs[:len(valid_recall20)], valid_recall20, label='Valid Recall@20', marker='o')
plt.plot(epochs[:len(valid_ndcg20)], valid_ndcg20, label='Valid NDCG@20', marker='x')

# Recall@20 및 NDCG@20 그래프 (Test)
plt.plot(epochs[:len(test_recall20)], test_recall20, label='Test Recall@20', marker='s')
plt.plot(epochs[:len(test_ndcg20)], test_ndcg20, label='Test NDCG@20', marker='^')

plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.title('Amazon-Book Recall@20 and NDCG@20 Training Curve')
plt.legend()
plt.grid(True)

# 이미지 저장
output_path = './training_curve_amazon-book.png'  # 저장할 경로 및 파일명
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 해상도(dpi)와 여백 설정
print(f"그래프가 {output_path}에 저장되었습니다.")

# 그래프 출력
plt.tight_layout()
plt.show()
