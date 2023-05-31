# 時相深層展開（TDU）（※1など） による時変比例ゲインの最適化
時相深層展開（※1など）の考え方を，時変の比例ゲイン列の最適化に応用してみた例です．  
PyTorch ※2の練習も兼ねて作成したサンプルコードです．

## 概要
<制御対象>
	```
	y[k+1] = a y[k] + b u[k]
	```
ここでスカラーパラメータa，bは既知とし，aは不安定な数値に設定する．

<制御器>
	```
	u[k] = f[k] ( ym[k] -y[k] )
	```

制御器中の時変ゲインf[k]の時系列ベクトル F := [ f[0], f[1], ... , f[N] ]^T を深層展開により最適化することを目指す．
初期ゲイン列はすべて1とする．

<損失関数>
	```
	J = norm( Ym -Y ) ^2
	```

ここで Ym := [ ym[0], ym[1], ..., ym[N] ]^T は目標値の時系列ベクトル，Y := [ y[0], y[1], ..., y[N] ]^T はプラント出力の時系列ベクトルである．

`main.py`を実行すると，上記損失関数を用いた勾配法が既定の回数実行される．

Figure.1 は初期ゲイン列と最適化ゲイン列を用いたプラント入出力結果を示す．
Figure.1 より，初期ゲインでは出力が発散しているようすがわかり，最適化後には安定化できていることがわかる．

またFigure.2 は学習中の損失関数の推移と，閉ループ系の固有値の絶対値の最大値（各ステップにおける）の推移をそれぞれ表す．
固有値の最大値ははじめ1を超えている（不安定）ものの，学習により速やかに安定範囲内に調整されていることがわかる．

## 参考文献
1. 時相深層展開を用いたモデル予測制御の多重振り子系に対する有効性の検証, 相澤純平, 小蔵正輝, 岸田昌子, 若宮直紀, システム制御情報学会論文誌, Vol.36, No.4, pp.91-98, (2023), https://cir.nii.ac.jp/crid/1520858851030340864
2. PyTorch実践入門 ディープラーニングの基礎から実装へ, Eli Stevens, Luca Antiga, Thomas Viehmann, マイナビ出版, (2021), https://book.mynavi.jp/ec/products/detail/id=120263