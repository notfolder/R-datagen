# 人工データ生成ノート

## 生成する人工データの種類1: 独立変数から説明できるいくつかの分布のYを生成する

10000行の独立変数1000個について線形モデルで説明できるYを下記分布で生成  
同じ変数Xについて、それぞれ分布の違うYを生成する

- 正規分布
- ポアソン分布
- 混合正規分布

最初10個の変数について、(10,9,8,..,1)と大きい係数、次の90個の変数について0.1、それ以外はYに関係しないものとする

上記独立変数のうち、21〜30までにX1〜X10の0.8掛けの共線性をもった変数を上書きして、
上記それぞれの分布のYを生成、[data-multicol.zip]として生成する


[data-independent.zip](https://zenodo.org/record/5151404)
