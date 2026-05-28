# Quantum AI: 핸즈온 양자 머신러닝 with Qiskit
#### 양자 신경망·양자 커널·하이브리드 모델을 처음부터 배포까지 핵심 가이드 

<img src="https://beat-by-wire.gitbook.io/beat-by-wire/~gitbook/image?url=https%3A%2F%2F3055094660-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FYzxz4QeW9UTrhrpWwKiQ%252Fuploads%252FtjpNaObid03Wcro3tCoD%252FQuantum-AI-Qiskit.png%3Falt%3Dmedia%26token%3D698b276c-2b3b-46b5-a262-42887b1257a6&width=300&dpr=3&quality=100&sign=f0980412&sv=2" width="500" height="707"/>



## 책 소개

양자 컴퓨팅을 처음 진지하게 다뤄본 사람들이 공통적으로 경험하는 순간이 있다. Bloch 구 위에서 |0⟩이 H 게이트 하나로 |+⟩으로 옮겨가는 모습을 보거나, Bell 상태가 두 큐비트를 분리할 수 없는 하나의 양자 상관으로 묶는 장면을 마주하며 "이 회로가 머신러닝의 패턴을 학습할 수 있을까"라는 질문을 품는 순간이다. 양자 컴퓨팅은 분명 놀랍다. 2ⁿ 차원 상태 공간을 n 큐비트로 표현하고, 측정의 본질적 확률성으로 통계를 만들며, 얽힘으로 고전 정보 이론을 넘어서는 능력은 불과 몇 년 전만 해도 학계의 추상 주제에 머물러 있었다.

문제는 그 다음이다. 양자 회로는 작은 예제에서는 우아하지만, 막상 머신러닝에 적용하려 하면 Feature Map의 인코딩 선택, Ansatz의 표현력, 옵티마이저의 noisy 기울기, Barren Plateau 같은 문제들이 쏟아진다. 이론은 추상적이고, 양자 회로와 머신러닝을 잇는 다리는 생각보다 복잡하다. 많은 실무자들이 바로 그 지점에서 멈춘다. 본 책은 그 지점을 통과하기 위해 썼다.

본 책은 양자 컴퓨팅과 머신러닝이 결합하여 분류·회귀·생성·차원축소를 수행하는 양자 머신러닝(Quantum Machine Learning, QML)의 원리와 방법을 이론과 실전 모두에서 체계적으로 다룬다. 단순히 알고리즘을 나열하는 것을 넘어, 각각의 접근법이 왜 등장했고, 어떤 문제를 해결하며, 본 책의 메인 도구인 Qiskit Machine Learning 1.x · PyTorch의 어떤 클래스와 연결되는지를 함께 짚는다.

지금 양자 ML의 시대 한가운데 서 있다. IBM Heron 133큐비트, Quantinuum H2, Atom Computing 1180큐비트 같은 NISQ 하드웨어가 매년 새로운 규모를 갱신하고, Liu et al. (2021)이 양자 커널의 quadratic speedup을 이론적으로 증명하며, Schuld (2021)가 "양자 ML은 본질적으로 커널 방법"임을 밝힌 지금, 양자 컴퓨팅과 머신러닝의 결합은 학계의 호기심을 넘어 실무 도구로 자리 잡고 있다.

그러나 양자 회로 단독으로는 작은 데이터셋의 분류조차 어렵다. Feature Map · Ansatz · 옵티마이저 · 측정 전략이 정교하게 조합되어야 비로소 학습이 작동한다. 그 조합의 원리와 트레이드오프를 손으로 직접 짚어볼 수 있도록 본 책은 모든 코드를 Qiskit 1.x와 PyTorch 2.x 위에서 실제로 동작하는 형태로 작성했다.

알고리즘의 이름을 아는 것과, 그 작동 원리를 손으로 직접 이해하고 설계해본 것은 전혀 다른 역량이다. 본 책을 마칠 때쯤이면, 새로운 양자 ML 논문을 읽고 그 설계 원리를 꿰뚫어 볼 수 있는 눈과, 자신의 데이터에 양자 회로를 직접 결합할 수 있는 토대가 만들어져 있을 것이다.


## 목 차

저자 소개
Table of Contents (목차)
서문: 들어가며
프롤로그: Quantum Computing — 수학적 기초

Part 1. QML(Quantum Machine Learning) 들어가기 전에
Chapter 1 — 양자 컴퓨팅 입문
Chapter 2 — Qiskit 개발 환경 설정
Chapter 3 — 머신러닝 핵심 복습

Part 2. 양자 신경망(Quantum Neural Networks)
Chapter 4 — 양자 신경망의 기초
Chapter 5 — 양자 신경망 분류기와 회귀기
Chapter 6 — 실전 데이터셋으로 양자 모델 학습하기

Part 3. 양자 커널 방법(Quantum Kernel Methods)
Chapter 7 — 양자 커널의 이론과 실습
Chapter 8 — 양자 커널을 이용한 비지도학습
Chapter 9 — Pegasos QSVC — 대규모 데이터를 위한 양자 SVM
Chapter 10 — 양자 커널 트레이너 — 커널 자체를 학습한다

Part 4. PyTorch 하이브리드 양자-고전 모델
Chapter 11 — Torch Connector — PyTorch와 양자 회로 통합
Chapter 12 — 양자 GAN — 생성적 적대 신경망

Part 5. 고급 양자 머신러닝 아키텍처
Chapter 13 — 양자 합성곱 신경망 (QCNN)
Chapter 14 — 양자 오토인코더 (Quantum Autoencoder)
Chapter 15 — 양자 베이지안 추론 (Quantum Bayesian Inference)

Part 6. Qiskit 운영과 최적화
Chapter 16 — Qiskit 모델 저장, 로딩, 연속 학습
Chapter 17 — 양자 신경망의 유효 차원 분석
Chapter 18 — 양자 모델의 성능 최적화 전략
Chapter 19 — 실제 양자 하드웨어에서의 실행

Part 7. QML 종합 프로젝트
Chapter 20 — 양자 ML 프로젝트
Chapter 21 — 프로젝트 I : 양자 머신러닝 파이프라인 구축
Chapter 22 — 프로젝트 II — PyTorch Connector 양자-고전 하이브리드

에필로그 — 양자 머신러닝의 현재와 미래


## E-Book 구매

- Yes24: https://www.yes24.com/product/goods/190241997
- 교보문고: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000013047392
- 알라딘: https://aladin.kr/p/lS1cp

## Github 코드: 

https://github.com/no-wave/quantum-ai-qiskit






