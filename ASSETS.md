# Модели и ресурсы

## Robust Video Matting

Авторы: Shanchuan Lin, Linjie Yang, Imran Saleemi, Soumyadip Sengupta.

- [Исходный проект и архитектура](https://github.com/PeterL1n/RobustVideoMatting).
- [Описание инференса](https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/inference.md).
- [Веса MobileNetV3 FP32, официальный релиз v1.0.0](https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx).
- Файл: `models/rvm_mobilenetv3_fp32.onnx`.
- SHA-256: `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828`.

RVM распространяется под GPL-3.0. Текст лицензии сохранён в [LICENSE](LICENSE).

## MediaPipe

[Google MediaPipe Selfie Segmentation](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/selfie_segmentation.md), версия 0.10.21. Веса входят в Python-пакет. Лицензия проекта — [Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE).

## Фоны и демонстрация

`backgrounds/office.png` — статичный фон комнаты. Три градиентных фона вычисляются в `backgrounds.py`.

`demo/demo.mp4` — 20-секундная запись работы приложения с веб-камерой на Apple M4 Pro, сделанная 12 октября 2026 года с восемью потоками RVM. Измерения этой записи сохранены в `results/demo.json`. `demo/background_removal.jpg` — кадр около 17-й секунды из этой же записи. Модель обрабатывает исходные кадры с камеры; фон подставляется после инференса.
