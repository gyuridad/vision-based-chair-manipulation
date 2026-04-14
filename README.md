Isaac Sim 로봇팔 의자 집기 영상
  - https://drive.google.com/file/d/1XZfXE9Lunah0PNjpQGUyQhGaWx0Rd4VG/view?usp=drive_link 

프로젝트 개요

  - 이 프로젝트는 ROS 2와 MoveIt IK/FK를 활용해 카메라로 검출한 의자의 3D 위치를 기준으로 로봇팔이 의자를 집고 들어올리는 파이프라인을 구현한 것입니다.
    기존 버전인 chair_grasp_moveit.py는 목표 자세 두 점에 대해 IK를 계산한 뒤 joint space 보간으로 접근했기 때문에,
    하강 구간에서 팔이 옆으로 휘거나 손목이 불안정하게 회전하는 문제가 있었습니다.
    이를 개선하기 위해 chair_grasp_moveit_vertical_move.py에서는 x, y를 고정하고
    z만 단계적으로 낮추면서 매 스텝 IK를 다시 계산하는 방식으로 접근 경로를 재설계했습니다.

기술 스택

  - ROS 2 rclpy
  - MoveIt GetPositionIK, GetPositionFK
  - TF2 TransformListener, Buffer
  - Python
  - NumPy
  - Isaac Sim / ROS 연동 환경
  - Panda 로봇팔 JointState 기반 제어

주요 기능

  - 카메라 detection 결과를 월드 좌표계로 변환
  - MoveIt IK/FK를 이용한 grasp pose 계산
  - top-down orientation 기반 grasp 자세 생성
  - pre-grasp, grasp, close, lift, retreat 단계 실행
  - x, y 고정 + z만 감소하는 vertical approach 실행
  - 직전 IK 해를 seed로 사용하는 연속적 IK 추적
  - detection lock, stale detection 무시, orientation tolerance 등 안정화 옵션 제공

아키텍처 개요

  1. Detection 수신
     의자 검출 결과를 /chair_detection_json에서 수신합니다.

  2. 좌표 변환
     카메라 좌표의 3D 점을 TF 또는 t_world_camera를 이용해 월드 좌표로 변환합니다.

  3. 목표 자세 생성
     grasp 위치, pre-grasp 위치, lift 위치, retreat 위치를 생성하고 end-effector orientation을 결정합니다.

  4. IK/FK 계산
     MoveIt IK/FK 서비스를 사용해 각 목표 자세에 대응하는 joint state를 계산합니다.

  5. trajectory 실행
     기존 방식은 두 IK 해 사이를 joint interpolation으로 연결했고, 개선 방식은 하강 구간에서 x, y를 유지한 채 z만 낮추며 IK를 연속적으로 다시 풉니다.

문제 해결 사례

  - 기존 chair_grasp_moveit.py에서는 접근 경로가 end-effector 기준 직선 하강이 아니었습니다.
    구체적으로는 pre_pos와 goal_pos 각각에 대해 IK를 한 번씩만 계산한 뒤,
    두 joint 해 사이를 선형 보간했기 때문에 실제 로봇팔은 “수직으로 내리는” 대신 “관절값 사이를 연결하는” 형태로 움직였습니다.
    그 결과:

      - 팔꿈치가 크게 꺾이거나
      - 손목이 불필요하게 회전하고
      - 하강 중 end-effector가 옆으로 새는 현상
      - 이 발생해 grasp 성공률이 낮았습니다.

    이 문제를 해결하기 위해 chair_grasp_moveit_vertical_move.py에서는 접근 구간을 다음처럼 변경했습니다.

      - 목표 x, y를 고정
      - orientation도 고정
      - z만 조금씩 감소
      - 각 step마다 IK를 다시 계산
      - 직전 IK 해를 다음 IK의 seed로 사용
      - 이 방식의 핵심은 “최종 자세 하나를 향해 크게 점프하는 IK”가 아니라,
        “현재 자세에서 아주 조금 아래로 내려가는 IK”를 연속적으로 푸는 것입니다.
        그래서 해가 갑자기 다른 branch로 튀는 현상이 줄고,
        end-effector가 더 자연스럽고 직관적인 수직 하강 경로를 따르게 되었습니다.

개선 효과

  - 내부 실험 기준으로 기존 chair_grasp_moveit.py는 IK 경로 불안정 문제로 grasp 성공률이 약 30% 수준이었지만,
    chair_grasp_moveit_vertical_move.py 적용 이후에는 약 90% 이상까지 향상되었습니다.
    이 개선은 단순히 파라미터 튜닝이 아니라, “IK를 어떤 방식으로 연결해서 실제 경로를 만들 것인가”를 바꾼 결과입니다.

노션 내용
  - https://www.notion.so/_1_2-32c5d0bd69638071ba71cb54025ca933?showMoveTo=true 
