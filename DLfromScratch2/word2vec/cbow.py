import sys
sys.path.append('..')
import numpy as np
from DLfromScratch2.common.layers import MatMul

#샘플 맥락 데이터
c0 = np.array([[1,0,0,0,0,0,0]])
c1 = np.array([[0,0,1,0,0,0,0]])

# 가중치 초기화
w_in = np.random.randn(7,3)
w_out = np.random.randn(3,7)

# 계층 생성
in_Layer0 = MatMul(w_in)
in_Layer1 = MatMul(w_in)
out_layer =MatMul(w_out)

# 순전파
h0 = in_Layer0.forword(c0)
h1 = in_Layer1.forword(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forword(h)

print(s)