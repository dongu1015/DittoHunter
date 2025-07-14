# Standard Library
import os
import random
import time
import warnings
from io import BytesIO
import glob

# Third Party - Core
import numpy as np
import cv2
from PIL import Image

# Third Party - ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from PIL import Image
from torchvision import models, transforms
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATv2Conv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import optuna
import warnings

warnings.filterwarnings('ignore')

# PyTorch 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 사용 디바이스: {device}")

# CNN-GAT 모델 클래스들
class CNNExtractor(nn.Module):
    """CNN 특징 추출기 (MobileNetV2 기반)"""
    def __init__(self, output_dim=64):
        super().__init__()
        # MobileNetV2 사전 훈련된 모델 사용
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features
        self.cnn = nn.Sequential(*list(base.children()))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1280, output_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.pool(self.cnn(x)).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        return self.proj(x)

class GATNet(nn.Module):
    """Graph Attention Network 딥페이크 검출기"""
    def __init__(self, in_channels=64, hidden_channels=64, out_channels=2, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATv2Conv(in_channels, hidden_channels, heads=heads, dropout=dropout, add_self_loops=False)
        self.gat2 = GATv2Conv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, add_self_loops=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_channels)
        )

    def forward(self, x, edge_index, batch):
        x, _ = self.gat1(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x, _ = self.gat2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        x = global_mean_pool(x, batch)
        return self.mlp(x)

class SimpleDeepfakeDetectorTrainer:
    def __init__(self, data_dir="data"):
        """딥페이크 검출 모델 학습기 초기화 (로컬 파일 기반)"""
        self.data_dir = data_dir
        self.fake_dir = os.path.join(data_dir, "fake")
        self.real_dir = os.path.join(data_dir, "real")
        
        # 데이터 디렉토리 확인
        if not os.path.exists(self.fake_dir):
            os.makedirs(self.fake_dir, exist_ok=True)
            print(f"📁 Created fake images directory: {self.fake_dir}")
        if not os.path.exists(self.real_dir):
            os.makedirs(self.real_dir, exist_ok=True)
            print(f"📁 Created real images directory: {self.real_dir}")
        
        # CNN 변환기 설정
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 표준화
        ])
        
        # CNN 특징 추출기 초기화
        self.cnn_extractor = CNNExtractor().to(device).eval()
        print("🧠 CNN 특징 추출기 초기화 완료")

    def fft_preprocess(self, img, size=(256, 256)):
        """FFT 전처리로 주파수 도메인 특징 추출"""
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        img = cv2.resize(img, size)
        fft_channels = []
        
        for i in range(3):  # RGB 각 채널
            fft = np.fft.fftshift(np.fft.fft2(img[:, :, i]))
            magnitude = 20 * np.log(np.abs(fft) + 1e-8)
            fft_channels.append(magnitude)
        
        fft_img = np.stack(fft_channels, axis=2)
        return cv2.normalize(fft_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def simple_region_detection(self, img):
        """간단한 지역 감지 (얼굴/배경)"""
        h, w = img.shape[:2]
        
        # OpenCV 얼굴 검출기 사용
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_mask = np.zeros((h, w), np.uint8)
        for (x, y, w_face, h_face) in faces:
            cv2.rectangle(face_mask, (x, y), (x + w_face, y + h_face), 255, -1)
        
        # 배경 마스크 (얼굴이 아닌 부분)
        bg_mask = np.ones((h, w), np.uint8) * 255
        bg_mask[face_mask == 255] = 0
        
        return face_mask, bg_mask

    def split_into_patches(self, img, patch_size=(32, 32)):
        """이미지를 패치로 분할"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fft_img = self.fft_preprocess(img_pil.resize((256, 256)))
        
        patches = []
        for y in range(0, 256, patch_size[0]):
            for x in range(0, 256, patch_size[1]):
                patch = fft_img[y:y+patch_size[0], x:x+patch_size[1]]
                if patch.shape[:2] == patch_size:
                    patches.append(patch)
        
        return patches, patch_size[0]

    def create_graph_from_image(self, img, label):
        """이미지에서 그래프 데이터 생성 (간소화 버전)"""
        try:
            # 1. 지역 마스크 생성 (간소화)
            face_mask, bg_mask = self.simple_region_detection(img)
            
            # 2. 패치 분할
            patches, patch_size = self.split_into_patches(img)
            rows, cols = 256 // patch_size, 256 // patch_size
            
            if len(patches) != rows * cols:
                return None
            
            # 3. 패치별 특징 추출
            features = []
            for patch in patches:
                patch_tensor = self.transform(Image.fromarray(patch)).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = self.cnn_extractor(patch_tensor).squeeze(0).cpu()
                features.append(feat)
            
            # 4. 간단한 그래프 엣지 생성 (4-연결 그리드)
            edge_index = []
            
            for i in range(rows):
                for j in range(cols):
                    current_idx = i * cols + j
                    
                    # 4-연결 그리드
                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            neighbor_idx = ni * cols + nj
                            edge_index.append([current_idx, neighbor_idx])
            
            # 5. 그래프 데이터 생성
            if not edge_index:
                # 엣지가 없으면 자기 자신과 연결
                edge_index = [[i, i] for i in range(len(features))]
            
            graph_data = Data(
                x=torch.stack(features),
                edge_index=torch.tensor(edge_index, dtype=torch.long).T.contiguous(),
                y=torch.tensor(label, dtype=torch.long)
            )
            
            return graph_data
            
        except Exception as e:
            print(f"❌ 그래프 생성 오류: {e}")
            return None

    def apply_data_augmentation(self, fake_graphs, real_graphs, augmentation_factor=2):
        """데이터 증강으로 학습 데이터 확장"""
        print(f"🔄 데이터 증강 시작 (증강 배수: {augmentation_factor}x)")
        
        original_fake_count = len(fake_graphs)
        original_real_count = len(real_graphs)
        
        # Fake 데이터 증강
        augmented_fake = []
        for _ in range(augmentation_factor - 1):  # 원본 제외하고 추가 생성
            for graph in fake_graphs:
                # 그래프 데이터에서 원본 특징 추출은 어려우므로, 
                # 동일한 그래프에 약간의 노이즈 추가
                augmented_graph = Data(
                    x=graph.x + torch.randn_like(graph.x) * 0.01,  # 작은 노이즈 추가
                    edge_index=graph.edge_index,
                    y=graph.y
                )
                augmented_fake.append(augmented_graph)
        
        # Real 데이터 증강
        augmented_real = []
        for _ in range(augmentation_factor - 1):
            for graph in real_graphs:
                augmented_graph = Data(
                    x=graph.x + torch.randn_like(graph.x) * 0.01,
                    edge_index=graph.edge_index,
                    y=graph.y
                )
                augmented_real.append(augmented_graph)
        
        # 증강된 데이터 추가
        fake_graphs.extend(augmented_fake)
        real_graphs.extend(augmented_real)
        
        print(f"✅ 데이터 증강 완료!")
        print(f"   Fake: {original_fake_count:,} → {len(fake_graphs):,}개 (+{len(augmented_fake):,})")
        print(f"   Real: {original_real_count:,} → {len(real_graphs):,}개 (+{len(augmented_real):,})")
        
        return fake_graphs, real_graphs

    def load_training_data(self, num_samples=None):
        """로컬 디렉토리에서 학습 데이터 로드"""
        
        # 로컬 파일 경로에서 이미지 수집
        fake_files = []
        real_files = []
        
        # fake 이미지 수집
        fake_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for ext in fake_extensions:
            fake_files.extend(glob.glob(os.path.join(self.fake_dir, ext)))
            fake_files.extend(glob.glob(os.path.join(self.fake_dir, ext.upper())))
        
        # real 이미지 수집
        real_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for ext in real_extensions:
            real_files.extend(glob.glob(os.path.join(self.real_dir, ext)))
            real_files.extend(glob.glob(os.path.join(self.real_dir, ext.upper())))
        
        print(f"📊 발견된 이미지:")
        print(f"   Fake: {len(fake_files):,}개")
        print(f"   Real: {len(real_files):,}개")
        
        if len(fake_files) < 5 or len(real_files) < 5:
            print(f"⚠️  Warning: 학습에 필요한 최소 데이터가 부족합니다.")
            print(f"   각 폴더에 최소 5개 이상의 이미지가 필요합니다.")
            print(f"   Fake 폴더: {self.fake_dir}")
            print(f"   Real 폴더: {self.real_dir}")
            return [], []
        
        # 샘플 수 제한 적용
        if num_samples is not None:
            fake_samples = min(num_samples // 2, len(fake_files))
            real_samples = min(num_samples // 2, len(real_files))
            fake_files = random.sample(fake_files, fake_samples)
            real_files = random.sample(real_files, real_samples)
            print(f"🎯 제한된 샘플 사용:")
            print(f"   Fake: {len(fake_files):,}개")
            print(f"   Real: {len(real_files):,}개")
        
        fake_graphs = []
        real_graphs = []
        
        success_count = 0
        total_attempts = len(fake_files) + len(real_files)
        
        print(f"📚 학습 데이터 생성 시작: {total_attempts:,}개 처리 예정")
        
        # Fake 이미지 처리
        print("🔴 Fake 이미지 처리 중...")
        for i, file_path in enumerate(fake_files):
            if i % 50 == 0:
                print(f"Fake 진행률: {i+1:,}/{len(fake_files):,} ({(i+1)/len(fake_files)*100:.1f}%)")
            
            try:
                # 이미지 로드
                img = cv2.imread(file_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # 그래프 데이터 생성
                graph = self.create_graph_from_image(img, label=0)  # fake = 0
                if graph is not None:
                    fake_graphs.append(graph)
                    success_count += 1
                    
            except Exception as e:
                continue
        
        # Real 이미지 처리
        print("🟢 Real 이미지 처리 중...")
        for i, file_path in enumerate(real_files):
            if i % 50 == 0:
                print(f"Real 진행률: {i+1:,}/{len(real_files):,} ({(i+1)/len(real_files)*100:.1f}%)")
            
            try:
                # 이미지 로드
                img = cv2.imread(file_path)
                if img is None:
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                
                # 그래프 데이터 생성
                graph = self.create_graph_from_image(img, label=1)  # real = 1
                if graph is not None:
                    real_graphs.append(graph)
                    success_count += 1
                    
            except Exception as e:
                continue
        
        print(f"\n✅ 데이터 로드 완료!")
        print(f"   처리된 이미지: {total_attempts:,}개")
        print(f"   성공한 그래프: {success_count:,}개")
        print(f"   Fake 그래프: {len(fake_graphs):,}개")
        print(f"   Real 그래프: {len(real_graphs):,}개")
        print(f"   성공률: {success_count/total_attempts*100:.1f}%")
        
        return fake_graphs, real_graphs

    def optimize_hyperparameters(self, train_loader, val_loader, n_trials=10):
        """Optuna를 사용한 하이퍼파라미터 최적화"""
        print(f"🔧 하이퍼파라미터 최적화 시작 ({n_trials}회 시도)")
        
        def objective(trial):
            # 하이퍼파라미터 후보
            hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
            heads = trial.suggest_categorical("heads", [1, 2, 4])
            dropout = trial.suggest_float("dropout", 0.0, 0.4)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
            
            # 모델 생성
            model = GATNet(
                in_channels=64,
                hidden_channels=hidden_channels,
                out_channels=2,
                heads=heads,
                dropout=dropout
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.CrossEntropyLoss()
            
            # 빠른 학습 (3 에포크)
            model.train()
            for epoch in range(3):
                for batch in train_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            # 검증
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    pred = out.argmax(dim=1)
                    correct += (pred == batch.y).sum().item()
                    total += batch.y.size(0)
            
            accuracy = correct / total
            return accuracy
        
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        print(f"🎯 최적화 완료! 최고 정확도: {study.best_value:.4f}")
        print(f"📋 최적 하이퍼파라미터:")
        for key, value in study.best_params.items():
            print(f"   {key}: {value}")
        
        return study.best_params

    def train_model(self, num_samples=None, epochs=50, use_optuna=True, use_data_augmentation=True, test_size=0.15):
        """딥페이크 검출 모델 학습 (전체 데이터셋 + 데이터 증강)"""
        print("🚀 딥페이크 검출 모델 학습 시작! (전체 데이터셋)")
        
        # 1. 데이터 로드
        fake_graphs, real_graphs = self.load_training_data(num_samples)
        
        # 2. 데이터 증강 적용
        if use_data_augmentation:
            fake_graphs, real_graphs = self.apply_data_augmentation(fake_graphs, real_graphs, augmentation_factor=3)
        
        # 3. 데이터 분할
        fake_train, fake_test = train_test_split(fake_graphs, test_size=test_size, random_state=42)
        real_train, real_test = train_test_split(real_graphs, test_size=test_size, random_state=42)
        
        train_data = fake_train + real_train
        test_data = fake_test + real_test
        
        # 데이터 셔플
        random.shuffle(train_data)
        random.shuffle(test_data)
        
        # 검증 데이터 분할 (학습 데이터의 15%)
        val_size = max(50, len(train_data) // 7)  # 최소 50개, 최대 전체의 1/7
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]
        
        # 데이터로더 생성 (배치 크기 증가)
        batch_size = min(8, len(train_data) // 10)  # 적응적 배치 크기
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        print(f"\n📊 최종 데이터 분할:")
        print(f"   학습: {len(train_data):,}개")
        print(f"   검증: {len(val_data):,}개") 
        print(f"   테스트: {len(test_data):,}개")
        print(f"   배치 크기: {batch_size}")
        
        # 4. 하이퍼파라미터 최적화
        if use_optuna and len(train_data) > 100:
            print(f"\n🔧 하이퍼파라미터 최적화 ({len(train_data):,}개 학습 데이터)")
            best_params = self.optimize_hyperparameters(train_loader, val_loader, n_trials=20)
        else:
            best_params = {
                "hidden_channels": 128,
                "heads": 4,
                "dropout": 0.15,
                "lr": 0.002,
                "weight_decay": 1e-5
            }
            print("🔧 기본 하이퍼파라미터 사용 (대용량 데이터에 최적화)")
        
        # 5. 최종 모델 학습
        print(f"\n🎯 최종 모델 학습 ({epochs} 에포크, {len(train_data):,}개 학습 데이터)")
        
        model = GATNet(
            in_channels=64,
            hidden_channels=best_params["hidden_channels"],
            out_channels=2,
            heads=best_params["heads"],
            dropout=best_params["dropout"]
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=best_params["lr"],
            weight_decay=best_params["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)
        
        best_acc = 0.0
        patience_counter = 0
        max_patience = 15  # 대용량 데이터에서는 더 많은 patience
        
        train_start_time = time.time()
        
        for epoch in range(epochs):
            # 학습
            model.train()
            total_loss = 0
            batch_count = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y)
                optimizer.zero_grad()
                loss.backward()
                
                # 그래디언트 클리핑 (안정성 향상)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
                batch_count += 1
            
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            
            # 검증
            model.eval()
            val_correct = val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    out = model(batch.x, batch.edge_index, batch.batch)
                    loss = criterion(out, batch.y)
                    val_loss += loss.item()
                    
                    pred = out.argmax(dim=1)
                    val_correct += (pred == batch.y).sum().item()
                    val_total += batch.y.size(0)
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            scheduler.step(val_acc)
            
            # 최고 모델 저장
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'hyperparameters': best_params,
                    'epoch': epoch,
                    'accuracy': best_acc,
                    'training_samples': len(train_data)
                }, "best_deepfake_detector_full.pth")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 조기 종료
            if patience_counter >= max_patience:
                print(f"💤 조기 종료 (patience: {max_patience})")
                break
            
            # 진행 상황 출력 (5 에포크마다 또는 처음/마지막)
            if epoch % 5 == 0 or epoch == epochs - 1 or epoch < 3:
                elapsed = time.time() - train_start_time
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f} | "
                      f"Best: {best_acc:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
        
        # 6. 최종 평가
        print(f"\n📊 최종 평가 ({len(test_data):,}개 테스트 데이터)")
        checkpoint = torch.load("best_deepfake_detector_full.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 테스트 데이터 평가
        model.eval()
        y_true, y_pred, y_prob = [], [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                prob = F.softmax(out, dim=1)
                pred = out.argmax(dim=1)
                
                y_true.extend(batch.y.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())
                y_prob.extend(prob[:, 1].cpu().tolist())  # Real 클래스 확률
        
        # 성능 지표 출력
        test_acc = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i]) / len(y_true)
        
        try:
            auc_score = roc_auc_score(y_true, y_prob)
        except:
            auc_score = 0.0
        
        total_time = time.time() - train_start_time
        
        print(f"\n🎉 학습 완료!")
        print(f"   학습 데이터: {len(train_data):,}개")
        print(f"   최고 검증 정확도: {best_acc:.4f}")
        print(f"   테스트 정확도: {test_acc:.4f}")
        print(f"   AUC 점수: {auc_score:.4f}")
        print(f"   총 학습 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"   평균 에포크 시간: {total_time/epoch:.1f}초")
        
        print(f"\n📋 분류 보고서:")
        print(classification_report(y_true, y_pred, target_names=["Fake", "Real"], digits=4))
        
        print(f"\n🧩 혼동 행렬:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # 클래스별 정확도
        fake_acc = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        real_acc = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
        
        print(f"\n🎯 클래스별 정확도:")
        print(f"   Fake 검출 정확도: {fake_acc:.4f}")
        print(f"   Real 검출 정확도: {real_acc:.4f}")
        
        return model, best_acc, test_acc, auc_score

    def close(self):
        """리소스 정리"""
        print("✅ 학습 완료 - 리소스 정리")

def main():
    """메인 실행 함수"""
    print("=== 딥페이크 검출 CNN-GAT 모델 학습기 (로컬 파일 기반) ===")
    
    trainer = SimpleDeepfakeDetectorTrainer(data_dir="data")
    
    try:
        # 학습 설정 (전체 데이터셋 사용)
        num_samples = None  # None이면 전체 데이터셋 사용
        epochs = 50         # 더 많은 에포크
        use_optuna = True   # 하이퍼파라미터 최적화 사용
        use_data_augmentation = True  # 데이터 증강 사용
        
        print(f"📋 학습 설정:")
        print(f"   샘플 수: {'전체 데이터셋' if num_samples is None else num_samples}")
        print(f"   에포크: {epochs}")
        print(f"   하이퍼파라미터 최적화: {use_optuna}")
        print(f"   데이터 증강: {use_data_augmentation}")
        
        # 모델 학습 실행
        model, best_val_acc, test_acc, auc_score = trainer.train_model(
            num_samples=num_samples,
            epochs=epochs,
            use_optuna=use_optuna,
            use_data_augmentation=use_data_augmentation
        )
        
        print(f"\n🏆 최종 결과:")
        print(f"   검증 정확도: {best_val_acc:.4f}")
        print(f"   테스트 정확도: {test_acc:.4f}")
        print(f"   AUC 점수: {auc_score:.4f}")
        print(f"   모델 저장: best_deepfake_detector.pth")
        
    except Exception as e:
        print(f"❌ 학습 오류: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        trainer.close()

if __name__ == "__main__":
    main()
