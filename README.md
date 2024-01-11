## はじめに
- このフォルダのコードは2021年度修士卒業生Keisuke Kimuraの研究に用いられたものです。
- コードは基本的にすべてJupyter Notebook形式で書かれています。
- Python3.11で動作確認済みです。


## ファイル構成
### SourceOptimization
- スピーカ配置最適化に関するコード

|ファイル名|説明|
|---|---|
|FreeField.ipynb|自由空間シミュレーション| 
|Room.ipynb|残響環境シミュレーション|
|posCandidate.csv|スピーカ配置候補点の座標|
|figs|図表保存用フォルダ|

### ILDReproduction
- 振幅マッチングを利用した音場合成に関するコード

|ファイル名|説明|
|---|---|
|NumSim.ipynb|提案法と従来法の比較，時間領域駆動信号フィルタ生成など| 
|SubExp.ipynb|主観評価実験の分析|
|posSrc230.csv|230のスピーカ配置の座標|
|exp\_answer.csv<br>exp\_n\_list.npy<br>exp\_order.csv|主観評価実験の結果|
|calc\_data|駆動信号，HRTF，WMMの重みなど，計算済みデータ|
|drv\_sigs|32chスピーカアレイ再生用の駆動信号|
|sample_music|サンプル音源(MUSDB18-HQ)|
|figs|図表保存用フォルダ|

### wave\_func.py
- 平面波・球面波の音圧値、およびそれらの球波動関数展開係数を返す関数をまとめたもの

### sf\_util.py
- 音場のシミュレーションに有用な関数をまとめたもの

## HRTFについて
- Mesh2HRTFを用いて計算した。
- `EvaluationGrid_{x}_{y}.sofa`は(x/10,y/10,0)に頭部中心を置いた場合の各スピーカ，(2,0,0)の所望音源から両耳までのHRTF（10〜20000Hz，10Hz刻み）
- `EvaluationGrid_4_{y}_{1-4}.sofa`はその分割（Mesh2HRTFでまとめて計算できなかったため）（100〜20000Hz，100Hz刻み）
- Mesh2HRTFを使用する際にたまに起こる`Isolated node found!`のエラーは結局解決策が不明。ノードを分割して計算した。
