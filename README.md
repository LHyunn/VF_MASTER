# VF Generator On NDT images for welded pipes

VF Generator is a tool for generating virtual defects on non-destructive testing (NDT) images of welded pipes. It allows you to overlay various types of defects on the images. Please note that I do not have ownership of the data used for the research, so I cannot publicly share the images. 

>Currently supported types of defects: IP, PO, Scratch, Leftover, CT

##### `THE PROVIDED CODE IS WRITTEN BASED ON THE IMAGE SIZE OF 1256*1256 PROVIDED BY THE COMPANY FOR US.`

## Features

- Supports various types of defects such as IP, PO, CT, Scratch, Leftover
- Adjustable parameters such as padding, fade, sigma, points
- Supports parallel processing with multiple workers

## Requirements

- Python 3.x
- OpenCV
- NumPy
- Ray
- tqdm
- elasticdeform

## Usage

1. Install the required dependencies by running the following command:

```
pip install opencv-python numpy ray tqdm elasticdeform
```

2. Prepare the defect numpy arrays.
- Place the PO defect images in the Extracted_Flaw/PO directory.
- Place the Scratch defect images in the Extracted_Flaw/Scratch directory.
- Place the Leftover defect images in the Extracted_Flaw/Leftover directory.
- Place the CT defect images in the Extracted_Flaw/CT directory.

3. Run the script using the following command:

```
python vf_generator.py --data_path <input image path> --save_path <result save path> --flaw_type <defect type> [additional options]
```

Replace *INPUT IMAGE PATH* with the directory path containing the input images, *RESULT SAVE PATH* with the desired directory path to save the generated defect images, and *DEFECT TYPE* with one of the following: IP, PO, CT, Scratch, Leftover.

##### Additional options:

- --padding: Set the padding value for the defect area (default: -30)
- --fade: Set the fade value for the defect boundary (default: 50)
- --nums_worker: Set the number of parallel workers (default: 1)
- --seed: Set the seed value for random number generation (default: 42)
- --CT_margin: Set the margin value for CT defects (default: 50)
- --sigma: Set the sigma value for elastic deformation (default: 5)
- --points: Set the number of control points for elastic deformation (default: 4)

The generated defect images will be saved in the specified save_path directory, organized into subdirectories based on the defect type (Accept, Reject, Diff).

## References

Image inpainting technique: https://github.com/advimman/lama









# VF Generator On NDT images for welded pipes

VF Generator은 용접 강관의 비파괴 검사 이미지에 가상 결함을 생성하는 도구입니다. 다양한 종류의 결함을 합성할 수 있습니다. 연구를 진행한 데이터의 소유권을 갖고 있지 않아 이미지를 외부에 공개할 수 없음을 양해 부탁드립니다.

>현재 지원하는 결함의 종류 : IP, PO, Scratch, Leftover, CT

##### `해당 코드들은 저희에게 데이터를 제공해준 기업 측의 이미지 사이즈인 1256*1256을 기준으로 작성되었습니다. `

## 특징

- IP, PO, CT, Scratch, Leftover 등 다양한 종류의 결함 지원
- 패딩, 페이드, 시그마, 포인트 등 조절 가능한 매개변수
- 병렬 처리 지원으로 다중 워커 사용
## 요구 사항

- Python 3.x
- OpenCV
- NumPy
- Ray
- tqdm
- elasticdeform

## 사용 방법

1. 다음 명령을 사용하여 필요한 종속성을 설치합니다:


```
pip install opencv-python numpy ray tqdm elasticdeform

```

2. 결함 numpy array를 준비합니다.
- PO 결함 이미지를 Extracted_Flaw/PO 디렉토리에 배치합니다.
- Scratch 결함 이미지를 Extracted_Flaw/Scratch 디렉토리에 배치합니다.
- Leftover 결함 이미지를 Extracted_Flaw/Leftover 디렉토리에 배치합니다.
- CT 결함 이미지를 Extracted_Flaw/CT 디렉토리에 배치합니다.

3. 다음 명령을 사용하여 스크립트를 실행합니다:


```
python vf_generator.py --data_path <입력 이미지 경로> --save_path <결과 저장 경로> --flaw_type <결함 종류> [기타 옵션]
```
<입력 이미지 경로>를 입력 이미지가 있는 디렉토리 경로로 대체합니다. <결과 저장 경로>를 생성된 결함 이미지를 저장할 원하는 디렉토리 경로로 대체합니다. <결함 종류>는 IP, PO, CT, Scratch, Leftover 중 하나여야 합니다.

추가 옵션:

- --padding: 결함 영역에 대한 패딩 값을 설정합니다 (기본값: -30)
- --fade: 결함 경계에 대한 페이드 값을 설정합니다 (기본값: 50)
- --nums_worker: 병렬 워커 수를 설정합니다 (기본값: 1)
- --seed: 난수 생성을 위한 시드 값을 설정합니다 (기본값: 42)
- --CT_margin: CT 결함에 대한 여백 값을 설정합니다 (기본값: 50)
- --sigma: 탄성 변형을 위한 시그마 값을 설정합니다 (기본값: 5)
- --points: 탄성 변형을 위한 제어 포인트 수를 설정합니다 (기본값: 4)

생성된 결함 이미지는 지정한 save_path 디렉토리에 하위 디렉토리로 결함 종류에 따라 정리되어 저장됩니다 (Accept, Reject, Diff). 

### 참고문헌

이미지 인페인팅 기법 : https://github.com/advimman/lama
