# main_CIFAR_10.py
# CIFAR-10 데이터셋을 사용한 사전 학습된 모델과 적대적 공격 구현
# 작성자: 이두현
# 작성일: 2025-04-04

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.utils import to_categorical
import os
import time

# GPU 메모리 할당 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 랜덤 시드 고정 (재현 가능성)
np.random.seed(42)
tf.random.set_seed(42)

# CIFAR-10 클래스 이름
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_cifar10_data():
    """CIFAR-10 데이터셋을 로드하고 전처리하는 함수
    
    Returns:
        x_train, y_train, x_test, y_test: 전처리된 학습 및 테스트 데이터
    """
    # CIFAR-10 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 데이터 형태 출력
    print('학습 데이터 형태:', x_train.shape)
    print('학습 라벨 형태:', y_train.shape)
    print('테스트 데이터 형태:', x_test.shape)
    print('테스트 라벨 형태:', y_test.shape)
    
    # 0~1 범위로 정규화
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 라벨 원-핫 인코딩
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)
    
    # 라벨 형태 변경 (2D -> 1D)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    return x_train, y_train_categorical, y_train, x_test, y_test_categorical, y_test

def create_pretrained_model(input_shape=(32, 32, 3), num_classes=10):
    """사전 학습된 ResNet50 모델을 사용한 CIFAR-10 분류 모델 생성
    
    Args:
        input_shape: 입력 이미지 형태
        num_classes: 출력 클래스 수
        
    Returns:
        model: 컴파일된 모델
    """
    # 입력 형태 설정
    inputs = Input(shape=input_shape)
    
    # 입력 이미지를 ResNet50 입력 크기(224x224)에 맞게 조정
    # 참고: 실제로는 업샘플링하지 않고 그대로 사용해도 됩니다 (모델 구조가 적응됨)
    
    # 사전 학습된 ResNet50 모델 불러오기 (imagenet으로 학습된 가중치 사용)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
    
    # 특징 추출 후 분류기 추가
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # 최종 모델 생성
    model = Model(inputs=inputs, outputs=predictions)
    
    # 모델 컴파일
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(x_train, y_train, x_test, y_test, model_path="cifar10_resnet50_model.h5"):
    """모델을 학습하는 함수
    
    Args:
        x_train: 학습 데이터
        y_train: 학습 라벨 (원-핫 인코딩)
        x_test: 테스트 데이터
        y_test: 테스트 라벨 (원-핫 인코딩)
        model_path: 모델 저장 경로
        
    Returns:
        model: 학습된 모델
    """
    # 이미 학습된 모델이 있는 경우 로드
    if os.path.exists(model_path):
        print(f"이미 학습된 모델을 불러옵니다: {model_path}")
        model = load_model(model_path)
        return model
    
    # 모델 생성
    model = create_pretrained_model()
    
    # 사전 학습된 ResNet50 레이어는 동결 (학습 X)
    for layer in model.layers[:-4]:  # 마지막 4개 레이어만 학습
        layer.trainable = False
    
    # 모델 요약 정보 출력
    model.summary()
    
    # 콜백 설정
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    
    # 모델 학습
    print("모델 학습을 시작합니다...")
    history = model.fit(
        x_train, y_train,
        batch_size=64,  # 작은 배치 사이즈 사용 (메모리 이슈)
        epochs=1000,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )
    
    # 학습 히스토리 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # 최종 모델 저장
    model.save(model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")
    
    # 테스트 데이터 평가
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print(f"테스트 정확도: {test_acc:.4f}")
    
    return model

# 적대적 공격 관련 함수들
def fgsm_targeted(model, x, target, eps=0.03):
    """Targeted FGSM (Fast Gradient Sign Method) 공격 구현
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        target: 타겟 클래스 (정수)
        eps: 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 타겟 원-핫 인코딩
    target_one_hot = tf.one_hot(target, 10)
    
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        prediction = model(x_adv)
        # 타겟 클래스의 확률을 최대화하는 방향으로 손실함수 정의
        loss = -tf.keras.losses.categorical_crossentropy(target_one_hot, prediction)
    
    # 입력에 대한 손실의 그레디언트 계산
    gradient = tape.gradient(loss, x_adv)
    
    # FGSM 방식으로 적대적 샘플 생성
    signed_grad = tf.sign(gradient)
    x_adv = x_adv - eps * signed_grad  # 타겟 방향으로 이동하기 위해 빼기 사용
    
    # [0, 1] 범위로 클리핑
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv.numpy()

def fgsm_untargeted(model, x, label, eps=0.03):
    """Untargeted FGSM (Fast Gradient Sign Method) 공격 구현
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        label: 원본 이미지의 실제 라벨 (정수)
        eps: 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 라벨 원-핫 인코딩
    label_one_hot = tf.one_hot(label, 10)
    
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        prediction = model(x_adv)
        # 실제 클래스의 확률을 최소화하는 방향으로 손실함수 정의
        loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
    
    # 입력에 대한 손실의 그레디언트 계산
    gradient = tape.gradient(loss, x_adv)
    
    # FGSM 방식으로 적대적 샘플 생성
    signed_grad = tf.sign(gradient)
    x_adv = x_adv + eps * signed_grad  # 원래 클래스에서 멀어지기 위해 더하기 사용
    
    # [0, 1] 범위로 클리핑
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv.numpy()

def pgd_targeted(model, x, target, k=10, eps=0.03, eps_step=0.01):
    """Targeted PGD (Projected Gradient Descent) 공격 구현
    
    Args:
        model: 타겟 모델
        x: 원본 이미지 (단일 이미지 또는 배치)
        target: 타겟 클래스 (정수)
        k: 반복 횟수
        eps: 최대 공격 강도
        eps_step: 각 단계의 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = x.copy()
    
    # 원본 이미지 범위 설정 (eps 반경 내)
    x_min = np.clip(x - eps, 0, 1)
    x_max = np.clip(x + eps, 0, 1)
    
    # 타겟 원-핫 인코딩
    target_one_hot = tf.one_hot(target, 10)
    
    for i in range(k):
        x_adv_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tensor)
            prediction = model(x_adv_tensor)
            # 타겟 클래스의 확률을 최대화하는 방향으로 손실함수 정의
            loss = -tf.keras.losses.categorical_crossentropy(target_one_hot, prediction)
        
        # 입력에 대한 손실의 그레디언트 계산
        gradient = tape.gradient(loss, x_adv_tensor)
        
        # 그레디언트 방향으로 한 스텝 이동
        signed_grad = tf.sign(gradient)
        x_adv = x_adv - eps_step * signed_grad.numpy()  # 타겟 방향으로 이동하기 위해 빼기 사용
        
        # 원본 이미지의 eps 범위 내로 제한 (투영)
        x_adv = np.clip(x_adv, x_min, x_max)
        
        # [0, 1] 범위로 추가 클리핑
        x_adv = np.clip(x_adv, 0, 1)
    
    return x_adv

def pgd_untargeted(model, x, label, k=10, eps=0.03, eps_step=0.01):
    """Untargeted PGD (Projected Gradient Descent) 공격 구현
    
    Args:
        model: 타겟 모델
        x: 원본 이미지 (단일 이미지 또는 배치)
        label: 원본 이미지의 실제 라벨 (정수)
        k: 반복 횟수
        eps: 최대 공격 강도
        eps_step: 각 단계의 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = x.copy()
    
    # 원본 이미지 범위 설정 (eps 반경 내)
    x_min = np.clip(x - eps, 0, 1)
    x_max = np.clip(x + eps, 0, 1)
    
    # 라벨 원-핫 인코딩
    label_one_hot = tf.one_hot(label, 10)
    
    for i in range(k):
        x_adv_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tensor)
            prediction = model(x_adv_tensor)
            # 실제 클래스의 확률을 최소화하는 방향으로 손실함수 정의
            loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
        
        # 입력에 대한 손실의 그레디언트 계산
        gradient = tape.gradient(loss, x_adv_tensor)
        
        # 그레디언트 방향으로 한 스텝 이동
        signed_grad = tf.sign(gradient)
        x_adv = x_adv + eps_step * signed_grad.numpy()  # 원래 클래스에서 멀어지기 위해 더하기 사용
        
        # 원본 이미지의 eps 범위 내로 제한 (투영)
        x_adv = np.clip(x_adv, x_min, x_max)
        
        # [0, 1] 범위로 추가 클리핑
        x_adv = np.clip(x_adv, 0, 1)
    
    return x_adv

def visualize_attack_results(model, x_orig, x_adv, true_label, attack_name, target=None):
    """적대적 공격 결과를 시각화하는 함수
    
    Args:
        model: 타겟 모델
        x_orig: 원본 이미지
        x_adv: 적대적 이미지
        true_label: 실제 라벨
        attack_name: 공격 이름
        target: 타겟 클래스 (targeted 공격에만 사용)
    """
    # 예측
    orig_pred = model.predict(np.expand_dims(x_orig, axis=0))[0]
    adv_pred = model.predict(np.expand_dims(x_adv, axis=0))[0]
    
    orig_class = np.argmax(orig_pred)
    adv_class = np.argmax(adv_pred)
    
    # 이미지 시각화
    plt.figure(figsize=(12, 4))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(x_orig)
    plt.title(f"Original: {cifar10_classes[orig_class]}\n(True: {cifar10_classes[true_label]})")
    plt.axis('off')
    
    # 적대적 이미지
    plt.subplot(1, 3, 2)
    plt.imshow(x_adv)
    if target is not None:
        plt.title(f"Adversarial: {cifar10_classes[adv_class]}\n(Target: {cifar10_classes[target]})")
    else:
        plt.title(f"Adversarial: {cifar10_classes[adv_class]}")
    plt.axis('off')
    
    # 차이 이미지
    plt.subplot(1, 3, 3)
    diff = np.abs(x_adv - x_orig)
    # 차이를 강조하기 위해 스케일 조정
    plt.imshow(diff * 5)  # 차이를 5배 강조
    plt.title("Perturbation (x5)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.suptitle(f"{attack_name} Attack", fontsize=16)
    plt.tight_layout()
    
    # 결과 저장 디렉토리 생성
    os.makedirs("attack_results", exist_ok=True)
    plt.savefig(f"attack_results/{attack_name}_example.png")
    
    # 예측 결과 출력
    print(f"\n{attack_name} Attack 결과")
    print(f"원본 이미지 예측: {cifar10_classes[orig_class]} (신뢰도: {orig_pred[orig_class]:.4f})")
    if target is not None:
        print(f"타겟 클래스: {cifar10_classes[target]}")
    print(f"적대적 이미지 예측: {cifar10_classes[adv_class]} (신뢰도: {adv_pred[adv_class]:.4f})")
    print(f"원본 이미지 라벨: {cifar10_classes[true_label]}")
    
    # 예측 확률 분포 비교
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(10), orig_pred)
    plt.xticks(range(10), cifar10_classes, rotation=45)
    plt.ylim([0, 1])
    plt.title("Original Prediction")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), adv_pred)
    plt.xticks(range(10), cifar10_classes, rotation=45)
    plt.ylim([0, 1])
    plt.title("Adversarial Prediction")
    
    plt.tight_layout()
    plt.savefig(f"attack_results/{attack_name}_predictions.png")

    # L2 거리 계산
    l2_distance = np.sqrt(np.sum((x_orig - x_adv) ** 2))
    print(f"L2 거리: {l2_distance:.6f}")
    
    # L∞ 거리 계산
    linf_distance = np.max(np.abs(x_orig - x_adv))
    print(f"L∞ 거리: {linf_distance:.6f}")
    
    # 공격 성공 여부
    if target is not None:
        success = (adv_class == target)
        print(f"공격 성공 여부 (타겟={cifar10_classes[target]}): {'성공' if success else '실패'}")
    else:
        success = (adv_class != true_label)
        print(f"공격 성공 여부 (원본 클래스에서 이탈): {'성공' if success else '실패'}")

def main():
    """메인 함수"""
    print("CIFAR-10 데이터 로드 중...")
    x_train, y_train_cat, y_train, x_test, y_test_cat, y_test = load_cifar10_data()
    
    print("\n사전 학습된 모델 로드 또는 학습 중...")
    model_path = "cifar10_resnet50_model.h5"
    model = train_model(x_train, y_train_cat, x_test, y_test_cat, model_path)
    
    print("\n학습된 모델로 테스트 데이터 평가 중...")
    loss, accuracy = model.evaluate(x_test, y_test_cat, verbose=1)
    print(f"테스트 정확도: {accuracy:.4f}")
    
    print("\n적대적 공격 예시 생성 중...")
    
    # 테스트 이미지 선택
    idx = np.random.randint(0, len(x_test))
    x = x_test[idx]
    true_label = y_test[idx]
    
    # 타겟 클래스 선택 (원래 클래스와 다르게)
    available_targets = [i for i in range(10) if i != true_label]
    target = np.random.choice(available_targets)
    
    print(f"\n선택된 이미지 인덱스: {idx}")
    print(f"실제 라벨: {true_label} ({cifar10_classes[true_label]})")
    print(f"타겟 라벨: {target} ({cifar10_classes[target]})")
    
    # 원본 이미지 예측
    orig_pred = model.predict(np.expand_dims(x, axis=0))[0]
    orig_class = np.argmax(orig_pred)
    print(f"원본 이미지 예측: {orig_class} ({cifar10_classes[orig_class]}, 신뢰도: {orig_pred[orig_class]:.4f})")
    
    # 1. Targeted FGSM 공격
    print("\nTargeted FGSM 공격 수행 중...")
    start_time = time.time()
    x_adv_fgsm_t = fgsm_targeted(model, np.expand_dims(x, axis=0), target, eps=0.03)
    print(f"공격 소요 시간: {time.time() - start_time:.2f}초")
    visualize_attack_results(model, x, x_adv_fgsm_t[0], true_label, "Targeted_FGSM", target)
    
    # 2. Untargeted FGSM 공격
    print("\nUntargeted FGSM 공격 수행 중...")
    start_time = time.time()
    x_adv_fgsm_u = fgsm_untargeted(model, np.expand_dims(x, axis=0), true_label, eps=0.03)
    print(f"공격 소요 시간: {time.time() - start_time:.2f}초")
    visualize_attack_results(model, x, x_adv_fgsm_u[0], true_label, "Untargeted_FGSM")
    
    # 3. Targeted PGD 공격
    print("\nTargeted PGD 공격 수행 중...")
    start_time = time.time()
    x_adv_pgd_t = pgd_targeted(model, np.expand_dims(x, axis=0), target, k=10, eps=0.03, eps_step=0.005)
    print(f"공격 소요 시간: {time.time() - start_time:.2f}초")
    visualize_attack_results(model, x, x_adv_pgd_t[0], true_label, "Targeted_PGD", target)
    
    # 4. Untargeted PGD 공격
    print("\nUntargeted PGD 공격 수행 중...")
    start_time = time.time()
    x_adv_pgd_u = pgd_untargeted(model, np.expand_dims(x, axis=0), true_label, k=10, eps=0.03, eps_step=0.005)
    print(f"공격 소요 시간: {time.time() - start_time:.2f}초")
    visualize_attack_results(model, x, x_adv_pgd_u[0], true_label, "Untargeted_PGD")
    
    print("\n적대적 공격 시연 완료!")

if __name__ == "__main__":
    main()