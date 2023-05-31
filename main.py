import numpy as np  # numpyのインポート
import torch        # pytorchのインポート
import matplotlib.pyplot as plt # matplotlibのインポート
#from matplotlib.ticker import ScalarFormatter

# =========================================
## 関数定義
# =========================================
# 制御対象のパラメータ
def plant_param():
    a = 1.9 # 不安定なパラメータに設定
    b = 0.3
    return a, b

# 制御対象のモデル
def plant( y, u ):
    param = plant_param()
    a = param[0]
    b = param[1]

    y1 = a * y + b * u

    return y1

# コントローラのモデル
def controller( ym, y, f ):
    u = f * ( ym -y )
    return u

# データ生成用関数
def simulation( ym, f, y0 ):
    N = len(f)
    y = torch.zeros(N)
    y[0] = y0
    u = torch.zeros(N)
    for i in range(N-1):
        u[i] = controller( ym[i], y[i], f[i] )
        y[i+1] = plant( y[i], u[i] )

    return y, u

# 損失関数
def loss_function( ym, y ):
    #J = ( ym -y )**2
    J = torch.linalg.vector_norm( ym -y )**2
    return J

# 全直列モデル
def series_model( y0, ym, f ):
    N = len(f)
    y = y0
    for i in range(N-1):
        u = controller( ym[i], y, f[i] )
        y = plant( y, u )

    return y

# 学習の実行関数
def trainingLoop( Ne, eta, w, y0, ym ):
    loss_h = np.zeros( Ne )
    pmax_h = np.zeros( Ne + 1 )

    pmax_h[0] = calcPoleMax( w )

    for i in range( 1, Ne + 1 ):
        # 勾配の初期化
        if w.grad is not None:
            w.grad.zero_()

        # モデルに基づく出力の計算
        output = simulation( ym, w, y0 )    # 順方向計算の実行
        y = output[0]
        # 損失関数の計算
        #loss  = loss_function( ym[-1], y[-1] ) # 規範出力の最終値とモデル出力の最終値を損失関数に使用
        loss  = loss_function( ym, y )  # 規範出力列とモデル出力列を損失関数に使用
        loss.backward() # 誤差逆伝搬の実行

        # 勾配法による重みの更新
        with torch.no_grad():
            w -= eta * w.grad

        # 最適化途中の表示
        if i % 10 == 0:
            print( 'Epoch %d, Loss %f' % ( i, float(loss)) )

        loss_h[i-1] = loss
        pmax_h[i] = calcPoleMax(w)

    return w, loss_h, pmax_h

# 極の計算関数
def calcPoleMax( f ):
    param = plant_param()
    a = param[0]
    b = param[1]
    #print( a -b * f[0:-1].detach().numpy())

    p = a -b * f[0:-2].detach().numpy()
    pmax = np.amax( np.abs( p ) )
    return pmax


# =========================================
## メイン部分
# =========================================
N = 11  # 制御ステップ回数

# 目標出力列
ym = torch.zeros(N)
# 初期ゲイン列
f  = np.ones(N)

# 初期値設定
y0 = 1

# 学習モデル出力の計算
#output = simulation( ym, f, y0 )

#print( y[9] )
#print(loss_function(ym[9],y[9]))

# torch.tensorとしてゲインを定義
w = torch.tensor( f, requires_grad=True)
#print(w)

# 損失関数の計算
#loss = loss_function( ym[9], series_model( y0, ym, w ) )
#print(loss)

# 誤差逆伝搬
#print(loss.backward())

# 勾配を計算
#print(w.grad)

## 繰り返し学習
eta = 0.0006    # ステップサイズの定義（試行錯誤で決定）
Ne  = 100       # 学習のエポック数（試行錯誤で決定）
trained_param = trainingLoop( Ne, eta, w, y0, ym )  # 学習の実行

w1 = trained_param[0]   # 最適化したゲイン
lf = trained_param[1]   # 学習中の損失関数の推移
pm = trained_param[2]   # 学習中の固有値の最大値の推移

print(f)    # 初期ゲインの表示
print(w1)   # 最適化後ゲインの表示


# =========================================
## 結果のプロット
# =========================================
ts = np.arange( N ) # 時間ステップ配列の生成

out_ini = simulation( ym, f, 1*y0 ) # 初期ゲインでの入出力の計算
out_opt = simulation( ym, w1, 1*y0 )# 最適化後のゲインでの入出力の計算

yini = out_ini[0]
yopt = out_opt[0].detach().numpy()  # torch.Tensor型からnumpy.array型への変換

uini = out_ini[1]
uopt = out_opt[1].detach().numpy()

# Figure 1
plt.figure()

# 出力の表示部分
plt.subplot(211)
plt.plot( ts, yini, '+-', label="Initial" )
plt.plot( ts, yopt, 'x-', label="Optimized" )
plt.ylim( -0.5, 3 * y0 )
plt.xlim( 0, 9 )
plt.grid()
plt.legend()
plt.xlabel("Time [step]")
plt.ylabel("Output [-]")

# 入力の表示部分
plt.subplot(212)
plt.plot( ts, uini, '+-', label="Initial" )
plt.plot( ts, uopt, 'x-', label="Optimized" )
plt.ylim( 2 * uopt[0] , 0.5 )
plt.xlim( 0, 9 )
plt.grid()
plt.legend()
plt.xlabel("Time [step]")
plt.ylabel("Input [-]")

# Figure 2
plt.figure()

# 損失関数の推移 
plt.subplot(211)
plt.plot( lf )
plt.yscale("log")
plt.xlim( left=0, right=20 )
plt.grid()
plt.xlabel("Epoch number, ( Max = %d )" % (Ne) )
plt.ylabel("Loss function")

# 閉ループ系の固有値の絶対値の最大値の推移
plt.subplot(212)
plt.plot( pm )
plt.xlim( left=0, right=20 )
plt.ylim( 0, 2 )
plt.grid()
plt.xlabel("Epoch number, ( Max = %d )" % (Ne) )
plt.ylabel("max( | a - b * f | )")

plt.show()
