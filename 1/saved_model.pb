��
�!�!
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
n
ClipByValue
t"T
clip_value_min"T
clip_value_max"T
output"T" 
Ttype:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
N
IsVariableInitialized
ref"dtype�
is_initialized
"
dtypetype�
:
Less
x"T
y"T
z
"
Ttype:
2	
,
Log
x"T
y"T"
Ttype:

2
.
Log1p
x"T
y"T"
Ttype:

2
#
	LogicalOr
x

y

z
�
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
�
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.8.02v1.8.0-0-g93bc2e2072��
p
dense_1_inputPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
m
dense_1/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_1/random_uniform/minConst*
valueB
 *�KF�*
dtype0*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
valueB
 *�KF?*
dtype0*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
T0*
dtype0*
_output_shapes

:*
seed2���*
seed���)
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
_output_shapes

:*
T0
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
_output_shapes

:*
T0
�
dense_1/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(
{
dense_1/kernel/readIdentitydense_1/kernel*
T0*!
_class
loc:@dense_1/kernel*
_output_shapes

:
Z
dense_1/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_1/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
q
dense_1/bias/readIdentitydense_1/bias*
T0*
_class
loc:@dense_1/bias*
_output_shapes
:
�
dense_1/MatMulMatMuldense_1_inputdense_1/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
\
activation_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������
m
dense_2/random_uniform/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
_
dense_2/random_uniform/minConst*
valueB
 *�Q�*
dtype0*
_output_shapes
: 
_
dense_2/random_uniform/maxConst*
valueB
 *�Q?*
dtype0*
_output_shapes
: 
�
$dense_2/random_uniform/RandomUniformRandomUniformdense_2/random_uniform/shape*
seed���)*
T0*
dtype0*
_output_shapes

:*
seed2���
z
dense_2/random_uniform/subSubdense_2/random_uniform/maxdense_2/random_uniform/min*
_output_shapes
: *
T0
�
dense_2/random_uniform/mulMul$dense_2/random_uniform/RandomUniformdense_2/random_uniform/sub*
T0*
_output_shapes

:
~
dense_2/random_uniformAdddense_2/random_uniform/muldense_2/random_uniform/min*
_output_shapes

:*
T0
�
dense_2/kernel
VariableV2*
dtype0*
_output_shapes

:*
	container *
shape
:*
shared_name 
�
dense_2/kernel/AssignAssigndense_2/kerneldense_2/random_uniform*
_output_shapes

:*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(
{
dense_2/kernel/readIdentitydense_2/kernel*
T0*!
_class
loc:@dense_2/kernel*
_output_shapes

:
Z
dense_2/ConstConst*
valueB*    *
dtype0*
_output_shapes
:
x
dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
dense_2/bias/AssignAssigndense_2/biasdense_2/Const*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
q
dense_2/bias/readIdentitydense_2/bias*
T0*
_class
loc:@dense_2/bias*
_output_shapes
:
�
dense_2/MatMulMatMulactivation_1/Tanhdense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
dense_2/BiasAddBiasAdddense_2/MatMuldense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
b
activation_2/SigmoidSigmoiddense_2/BiasAdd*
T0*'
_output_shapes
:���������
^
SGD/iterations/initial_valueConst*
value	B	 R *
dtype0	*
_output_shapes
: 
r
SGD/iterations
VariableV2*
shared_name *
dtype0	*
_output_shapes
: *
	container *
shape: 
�
SGD/iterations/AssignAssignSGD/iterationsSGD/iterations/initial_value*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: *
use_locking(
s
SGD/iterations/readIdentitySGD/iterations*
_output_shapes
: *
T0	*!
_class
loc:@SGD/iterations
Y
SGD/lr/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *���=
j
SGD/lr
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD/lr/AssignAssignSGD/lrSGD/lr/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(
[
SGD/lr/readIdentitySGD/lr*
T0*
_class
loc:@SGD/lr*
_output_shapes
: 
_
SGD/momentum/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
p
SGD/momentum
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
SGD/momentum/AssignAssignSGD/momentumSGD/momentum/initial_value*
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: 
m
SGD/momentum/readIdentitySGD/momentum*
T0*
_class
loc:@SGD/momentum*
_output_shapes
: 
\
SGD/decay/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	SGD/decay
VariableV2*
shape: *
shared_name *
dtype0*
_output_shapes
: *
	container 
�
SGD/decay/AssignAssign	SGD/decaySGD/decay/initial_value*
use_locking(*
T0*
_class
loc:@SGD/decay*
validate_shape(*
_output_shapes
: 
d
SGD/decay/readIdentity	SGD/decay*
T0*
_class
loc:@SGD/decay*
_output_shapes
: 
�
activation_2_targetPlaceholder*
dtype0*0
_output_shapes
:������������������*%
shape:������������������
v
activation_2_sample_weightsPlaceholder*#
_output_shapes
:���������*
shape:���������*
dtype0
a
loss/activation_2_loss/ConstConst*
valueB
 *���3*
dtype0*
_output_shapes
: 
a
loss/activation_2_loss/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
~
loss/activation_2_loss/subSubloss/activation_2_loss/sub/xloss/activation_2_loss/Const*
_output_shapes
: *
T0
�
$loss/activation_2_loss/clip_by_valueClipByValueactivation_2/Sigmoidloss/activation_2_loss/Constloss/activation_2_loss/sub*'
_output_shapes
:���������*
T0
c
loss/activation_2_loss/sub_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
loss/activation_2_loss/sub_1Subloss/activation_2_loss/sub_1/x$loss/activation_2_loss/clip_by_value*
T0*'
_output_shapes
:���������
�
loss/activation_2_loss/truedivRealDiv$loss/activation_2_loss/clip_by_valueloss/activation_2_loss/sub_1*
T0*'
_output_shapes
:���������
s
loss/activation_2_loss/LogLogloss/activation_2_loss/truediv*
T0*'
_output_shapes
:���������
�
/loss/activation_2_loss/logistic_loss/zeros_like	ZerosLikeloss/activation_2_loss/Log*
T0*'
_output_shapes
:���������
�
1loss/activation_2_loss/logistic_loss/GreaterEqualGreaterEqualloss/activation_2_loss/Log/loss/activation_2_loss/logistic_loss/zeros_like*
T0*'
_output_shapes
:���������
�
+loss/activation_2_loss/logistic_loss/SelectSelect1loss/activation_2_loss/logistic_loss/GreaterEqualloss/activation_2_loss/Log/loss/activation_2_loss/logistic_loss/zeros_like*'
_output_shapes
:���������*
T0
}
(loss/activation_2_loss/logistic_loss/NegNegloss/activation_2_loss/Log*
T0*'
_output_shapes
:���������
�
-loss/activation_2_loss/logistic_loss/Select_1Select1loss/activation_2_loss/logistic_loss/GreaterEqual(loss/activation_2_loss/logistic_loss/Negloss/activation_2_loss/Log*
T0*'
_output_shapes
:���������
�
(loss/activation_2_loss/logistic_loss/mulMulloss/activation_2_loss/Logactivation_2_target*
T0*0
_output_shapes
:������������������
�
(loss/activation_2_loss/logistic_loss/subSub+loss/activation_2_loss/logistic_loss/Select(loss/activation_2_loss/logistic_loss/mul*
T0*0
_output_shapes
:������������������
�
(loss/activation_2_loss/logistic_loss/ExpExp-loss/activation_2_loss/logistic_loss/Select_1*
T0*'
_output_shapes
:���������
�
*loss/activation_2_loss/logistic_loss/Log1pLog1p(loss/activation_2_loss/logistic_loss/Exp*
T0*'
_output_shapes
:���������
�
$loss/activation_2_loss/logistic_lossAdd(loss/activation_2_loss/logistic_loss/sub*loss/activation_2_loss/logistic_loss/Log1p*
T0*0
_output_shapes
:������������������
x
-loss/activation_2_loss/Mean/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
loss/activation_2_loss/MeanMean$loss/activation_2_loss/logistic_loss-loss/activation_2_loss/Mean/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
r
/loss/activation_2_loss/Mean_1/reduction_indicesConst*
valueB *
dtype0*
_output_shapes
: 
�
loss/activation_2_loss/Mean_1Meanloss/activation_2_loss/Mean/loss/activation_2_loss/Mean_1/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
�
loss/activation_2_loss/mulMulloss/activation_2_loss/Mean_1activation_2_sample_weights*#
_output_shapes
:���������*
T0
f
!loss/activation_2_loss/NotEqual/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
loss/activation_2_loss/NotEqualNotEqualactivation_2_sample_weights!loss/activation_2_loss/NotEqual/y*
T0*#
_output_shapes
:���������
�
loss/activation_2_loss/CastCastloss/activation_2_loss/NotEqual*

SrcT0
*#
_output_shapes
:���������*

DstT0
h
loss/activation_2_loss/Const_1Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/activation_2_loss/Mean_2Meanloss/activation_2_loss/Castloss/activation_2_loss/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
 loss/activation_2_loss/truediv_1RealDivloss/activation_2_loss/mulloss/activation_2_loss/Mean_2*
T0*#
_output_shapes
:���������
h
loss/activation_2_loss/Const_2Const*
valueB: *
dtype0*
_output_shapes
:
�
loss/activation_2_loss/Mean_3Mean loss/activation_2_loss/truediv_1loss/activation_2_loss/Const_2*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
O

loss/mul/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
[
loss/mulMul
loss/mul/xloss/activation_2_loss/Mean_3*
_output_shapes
: *
T0
|
training/SGD/gradients/ShapeConst*
_output_shapes
: *
_class
loc:@loss/mul*
valueB *
dtype0
�
 training/SGD/gradients/grad_ys_0Const*
_class
loc:@loss/mul*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
training/SGD/gradients/FillFilltraining/SGD/gradients/Shape training/SGD/gradients/grad_ys_0*
_output_shapes
: *
T0*
_class
loc:@loss/mul*

index_type0
�
(training/SGD/gradients/loss/mul_grad/MulMultraining/SGD/gradients/Fillloss/activation_2_loss/Mean_3*
_class
loc:@loss/mul*
_output_shapes
: *
T0
�
*training/SGD/gradients/loss/mul_grad/Mul_1Multraining/SGD/gradients/Fill
loss/mul/x*
T0*
_class
loc:@loss/mul*
_output_shapes
: 
�
Gtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Reshape/shapeConst*
_output_shapes
:*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
valueB:*
dtype0
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/ReshapeReshape*training/SGD/gradients/loss/mul_grad/Mul_1Gtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Reshape/shape*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
Tshape0*
_output_shapes
:
�
?training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/ShapeShape loss/activation_2_loss/truediv_1*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
out_type0*
_output_shapes
:
�
>training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/TileTileAtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Reshape?training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Shape*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*#
_output_shapes
:���������*

Tmultiples0*
T0
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Shape_1Shape loss/activation_2_loss/truediv_1*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
out_type0*
_output_shapes
:
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Shape_2Const*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
valueB *
dtype0*
_output_shapes
: 
�
?training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/ConstConst*
_output_shapes
:*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
valueB: *
dtype0
�
>training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/ProdProdAtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Shape_1?training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Const*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Const_1Const*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
valueB: *
dtype0*
_output_shapes
:
�
@training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Prod_1ProdAtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Shape_2Atraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
_output_shapes
: 
�
Ctraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Maximum/yConst*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
value	B :*
dtype0*
_output_shapes
: 
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/MaximumMaximum@training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Prod_1Ctraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Maximum/y*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
_output_shapes
: 
�
Btraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/floordivFloorDiv>training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/ProdAtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Maximum*
_output_shapes
: *
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3
�
>training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/CastCastBtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/floordiv*

SrcT0*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*
_output_shapes
: *

DstT0
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/truedivRealDiv>training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Tile>training/SGD/gradients/loss/activation_2_loss/Mean_3_grad/Cast*0
_class&
$"loc:@loss/activation_2_loss/Mean_3*#
_output_shapes
:���������*
T0
�
Btraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/ShapeShapeloss/activation_2_loss/mul*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*
out_type0*
_output_shapes
:
�
Dtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Shape_1Const*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*
valueB *
dtype0*
_output_shapes
: 
�
Rtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgsBtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/ShapeDtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1
�
Dtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/RealDivRealDivAtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/truedivloss/activation_2_loss/Mean_2*#
_output_shapes
:���������*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1
�
@training/SGD/gradients/loss/activation_2_loss/truediv_1_grad/SumSumDtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/RealDivRtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*
_output_shapes
:
�
Dtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/ReshapeReshape@training/SGD/gradients/loss/activation_2_loss/truediv_1_grad/SumBtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Shape*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*
Tshape0*#
_output_shapes
:���������
�
@training/SGD/gradients/loss/activation_2_loss/truediv_1_grad/NegNegloss/activation_2_loss/mul*#
_output_shapes
:���������*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1
�
Ftraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/RealDiv_1RealDiv@training/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Negloss/activation_2_loss/Mean_2*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*#
_output_shapes
:���������
�
Ftraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/RealDiv_2RealDivFtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/RealDiv_1loss/activation_2_loss/Mean_2*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*#
_output_shapes
:���������
�
@training/SGD/gradients/loss/activation_2_loss/truediv_1_grad/mulMulAtraining/SGD/gradients/loss/activation_2_loss/Mean_3_grad/truedivFtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/RealDiv_2*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*#
_output_shapes
:���������
�
Btraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Sum_1Sum@training/SGD/gradients/loss/activation_2_loss/truediv_1_grad/mulTtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*
_output_shapes
:
�
Ftraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Reshape_1ReshapeBtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Sum_1Dtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Shape_1*
T0*3
_class)
'%loc:@loss/activation_2_loss/truediv_1*
Tshape0*
_output_shapes
: 
�
<training/SGD/gradients/loss/activation_2_loss/mul_grad/ShapeShapeloss/activation_2_loss/Mean_1*
T0*-
_class#
!loc:@loss/activation_2_loss/mul*
out_type0*
_output_shapes
:
�
>training/SGD/gradients/loss/activation_2_loss/mul_grad/Shape_1Shapeactivation_2_sample_weights*
T0*-
_class#
!loc:@loss/activation_2_loss/mul*
out_type0*
_output_shapes
:
�
Ltraining/SGD/gradients/loss/activation_2_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs<training/SGD/gradients/loss/activation_2_loss/mul_grad/Shape>training/SGD/gradients/loss/activation_2_loss/mul_grad/Shape_1*
T0*-
_class#
!loc:@loss/activation_2_loss/mul*2
_output_shapes 
:���������:���������
�
:training/SGD/gradients/loss/activation_2_loss/mul_grad/MulMulDtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Reshapeactivation_2_sample_weights*
T0*-
_class#
!loc:@loss/activation_2_loss/mul*#
_output_shapes
:���������
�
:training/SGD/gradients/loss/activation_2_loss/mul_grad/SumSum:training/SGD/gradients/loss/activation_2_loss/mul_grad/MulLtraining/SGD/gradients/loss/activation_2_loss/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/activation_2_loss/mul
�
>training/SGD/gradients/loss/activation_2_loss/mul_grad/ReshapeReshape:training/SGD/gradients/loss/activation_2_loss/mul_grad/Sum<training/SGD/gradients/loss/activation_2_loss/mul_grad/Shape*
T0*-
_class#
!loc:@loss/activation_2_loss/mul*
Tshape0*#
_output_shapes
:���������
�
<training/SGD/gradients/loss/activation_2_loss/mul_grad/Mul_1Mulloss/activation_2_loss/Mean_1Dtraining/SGD/gradients/loss/activation_2_loss/truediv_1_grad/Reshape*
T0*-
_class#
!loc:@loss/activation_2_loss/mul*#
_output_shapes
:���������
�
<training/SGD/gradients/loss/activation_2_loss/mul_grad/Sum_1Sum<training/SGD/gradients/loss/activation_2_loss/mul_grad/Mul_1Ntraining/SGD/gradients/loss/activation_2_loss/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*-
_class#
!loc:@loss/activation_2_loss/mul
�
@training/SGD/gradients/loss/activation_2_loss/mul_grad/Reshape_1Reshape<training/SGD/gradients/loss/activation_2_loss/mul_grad/Sum_1>training/SGD/gradients/loss/activation_2_loss/mul_grad/Shape_1*
T0*-
_class#
!loc:@loss/activation_2_loss/mul*
Tshape0*#
_output_shapes
:���������
�
?training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/ShapeShapeloss/activation_2_loss/Mean*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
out_type0*
_output_shapes
:
�
>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/SizeConst*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
=training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/addAdd/loss/activation_2_loss/Mean_1/reduction_indices>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Size*
_output_shapes
: *
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1
�
=training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/modFloorMod=training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/add>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Size*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
_output_shapes
: 
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
valueB: 
�
Etraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/range/startConst*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
value	B : *
dtype0*
_output_shapes
: 
�
Etraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/range/deltaConst*
_output_shapes
: *0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
value	B :*
dtype0
�
?training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/rangeRangeEtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/range/start>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/SizeEtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/range/delta*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
_output_shapes
:*

Tidx0
�
Dtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Fill/valueConst*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/FillFillAtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Shape_1Dtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Fill/value*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*

index_type0*
_output_shapes
: 
�
Gtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/DynamicStitchDynamicStitch?training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/range=training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/mod?training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Shape>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Fill*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
N*#
_output_shapes
:���������
�
Ctraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Maximum/yConst*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
value	B :*
dtype0*
_output_shapes
: 
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/MaximumMaximumGtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/DynamicStitchCtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Maximum/y*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*#
_output_shapes
:���������
�
Btraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/floordivFloorDiv?training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/ShapeAtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Maximum*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*#
_output_shapes
:���������
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/ReshapeReshape>training/SGD/gradients/loss/activation_2_loss/mul_grad/ReshapeGtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/DynamicStitch*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
Tshape0*
_output_shapes
:
�
>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/TileTileAtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/ReshapeBtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/floordiv*

Tmultiples0*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
_output_shapes
:
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Shape_2Shapeloss/activation_2_loss/Mean*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
out_type0*
_output_shapes
:
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Shape_3Shapeloss/activation_2_loss/Mean_1*
_output_shapes
:*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
out_type0
�
?training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/ConstConst*
_output_shapes
:*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
valueB: *
dtype0
�
>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/ProdProdAtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Shape_2?training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Const*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Const_1Const*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
valueB: *
dtype0*
_output_shapes
:
�
@training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Prod_1ProdAtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Shape_3Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1
�
Etraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
value	B :
�
Ctraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Maximum_1Maximum@training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Prod_1Etraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Maximum_1/y*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
_output_shapes
: 
�
Dtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/floordiv_1FloorDiv>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/ProdCtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Maximum_1*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*
_output_shapes
: *
T0
�
>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/CastCastDtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/truedivRealDiv>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Tile>training/SGD/gradients/loss/activation_2_loss/Mean_1_grad/Cast*
T0*0
_class&
$"loc:@loss/activation_2_loss/Mean_1*#
_output_shapes
:���������
�
=training/SGD/gradients/loss/activation_2_loss/Mean_grad/ShapeShape$loss/activation_2_loss/logistic_loss*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*
out_type0*
_output_shapes
:
�
<training/SGD/gradients/loss/activation_2_loss/Mean_grad/SizeConst*.
_class$
" loc:@loss/activation_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
;training/SGD/gradients/loss/activation_2_loss/Mean_grad/addAdd-loss/activation_2_loss/Mean/reduction_indices<training/SGD/gradients/loss/activation_2_loss/Mean_grad/Size*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*
_output_shapes
: 
�
;training/SGD/gradients/loss/activation_2_loss/Mean_grad/modFloorMod;training/SGD/gradients/loss/activation_2_loss/Mean_grad/add<training/SGD/gradients/loss/activation_2_loss/Mean_grad/Size*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*
_output_shapes
: 
�
?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape_1Const*.
_class$
" loc:@loss/activation_2_loss/Mean*
valueB *
dtype0*
_output_shapes
: 
�
Ctraining/SGD/gradients/loss/activation_2_loss/Mean_grad/range/startConst*.
_class$
" loc:@loss/activation_2_loss/Mean*
value	B : *
dtype0*
_output_shapes
: 
�
Ctraining/SGD/gradients/loss/activation_2_loss/Mean_grad/range/deltaConst*.
_class$
" loc:@loss/activation_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
=training/SGD/gradients/loss/activation_2_loss/Mean_grad/rangeRangeCtraining/SGD/gradients/loss/activation_2_loss/Mean_grad/range/start<training/SGD/gradients/loss/activation_2_loss/Mean_grad/SizeCtraining/SGD/gradients/loss/activation_2_loss/Mean_grad/range/delta*.
_class$
" loc:@loss/activation_2_loss/Mean*
_output_shapes
:*

Tidx0
�
Btraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Fill/valueConst*.
_class$
" loc:@loss/activation_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
<training/SGD/gradients/loss/activation_2_loss/Mean_grad/FillFill?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape_1Btraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Fill/value*
_output_shapes
: *
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*

index_type0
�
Etraining/SGD/gradients/loss/activation_2_loss/Mean_grad/DynamicStitchDynamicStitch=training/SGD/gradients/loss/activation_2_loss/Mean_grad/range;training/SGD/gradients/loss/activation_2_loss/Mean_grad/mod=training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape<training/SGD/gradients/loss/activation_2_loss/Mean_grad/Fill*.
_class$
" loc:@loss/activation_2_loss/Mean*
N*#
_output_shapes
:���������*
T0
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Maximum/yConst*.
_class$
" loc:@loss/activation_2_loss/Mean*
value	B :*
dtype0*
_output_shapes
: 
�
?training/SGD/gradients/loss/activation_2_loss/Mean_grad/MaximumMaximumEtraining/SGD/gradients/loss/activation_2_loss/Mean_grad/DynamicStitchAtraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Maximum/y*#
_output_shapes
:���������*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean
�
@training/SGD/gradients/loss/activation_2_loss/Mean_grad/floordivFloorDiv=training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Maximum*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*
_output_shapes
:
�
?training/SGD/gradients/loss/activation_2_loss/Mean_grad/ReshapeReshapeAtraining/SGD/gradients/loss/activation_2_loss/Mean_1_grad/truedivEtraining/SGD/gradients/loss/activation_2_loss/Mean_grad/DynamicStitch*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*
Tshape0*
_output_shapes
:
�
<training/SGD/gradients/loss/activation_2_loss/Mean_grad/TileTile?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Reshape@training/SGD/gradients/loss/activation_2_loss/Mean_grad/floordiv*0
_output_shapes
:������������������*

Tmultiples0*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean
�
?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape_2Shape$loss/activation_2_loss/logistic_loss*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*
out_type0*
_output_shapes
:
�
?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape_3Shapeloss/activation_2_loss/Mean*.
_class$
" loc:@loss/activation_2_loss/Mean*
out_type0*
_output_shapes
:*
T0
�
=training/SGD/gradients/loss/activation_2_loss/Mean_grad/ConstConst*.
_class$
" loc:@loss/activation_2_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
�
<training/SGD/gradients/loss/activation_2_loss/Mean_grad/ProdProd?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape_2=training/SGD/gradients/loss/activation_2_loss/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean
�
?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Const_1Const*.
_class$
" loc:@loss/activation_2_loss/Mean*
valueB: *
dtype0*
_output_shapes
:
�
>training/SGD/gradients/loss/activation_2_loss/Mean_grad/Prod_1Prod?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Shape_3?training/SGD/gradients/loss/activation_2_loss/Mean_grad/Const_1*.
_class$
" loc:@loss/activation_2_loss/Mean*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
Ctraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Maximum_1/yConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@loss/activation_2_loss/Mean*
value	B :
�
Atraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Maximum_1Maximum>training/SGD/gradients/loss/activation_2_loss/Mean_grad/Prod_1Ctraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Maximum_1/y*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*
_output_shapes
: 
�
Btraining/SGD/gradients/loss/activation_2_loss/Mean_grad/floordiv_1FloorDiv<training/SGD/gradients/loss/activation_2_loss/Mean_grad/ProdAtraining/SGD/gradients/loss/activation_2_loss/Mean_grad/Maximum_1*
_output_shapes
: *
T0*.
_class$
" loc:@loss/activation_2_loss/Mean
�
<training/SGD/gradients/loss/activation_2_loss/Mean_grad/CastCastBtraining/SGD/gradients/loss/activation_2_loss/Mean_grad/floordiv_1*
_output_shapes
: *

DstT0*

SrcT0*.
_class$
" loc:@loss/activation_2_loss/Mean
�
?training/SGD/gradients/loss/activation_2_loss/Mean_grad/truedivRealDiv<training/SGD/gradients/loss/activation_2_loss/Mean_grad/Tile<training/SGD/gradients/loss/activation_2_loss/Mean_grad/Cast*
T0*.
_class$
" loc:@loss/activation_2_loss/Mean*0
_output_shapes
:������������������
�
Ftraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/ShapeShape(loss/activation_2_loss/logistic_loss/sub*
T0*7
_class-
+)loc:@loss/activation_2_loss/logistic_loss*
out_type0*
_output_shapes
:
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Shape_1Shape*loss/activation_2_loss/logistic_loss/Log1p*7
_class-
+)loc:@loss/activation_2_loss/logistic_loss*
out_type0*
_output_shapes
:*
T0
�
Vtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/ShapeHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0*7
_class-
+)loc:@loss/activation_2_loss/logistic_loss
�
Dtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/SumSum?training/SGD/gradients/loss/activation_2_loss/Mean_grad/truedivVtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@loss/activation_2_loss/logistic_loss*
_output_shapes
:
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/ReshapeReshapeDtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/SumFtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Shape*
T0*7
_class-
+)loc:@loss/activation_2_loss/logistic_loss*
Tshape0*0
_output_shapes
:������������������
�
Ftraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Sum_1Sum?training/SGD/gradients/loss/activation_2_loss/Mean_grad/truedivXtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@loss/activation_2_loss/logistic_loss*
_output_shapes
:
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Reshape_1ReshapeFtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Sum_1Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Shape_1*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@loss/activation_2_loss/logistic_loss*
Tshape0
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/ShapeShape+loss/activation_2_loss/logistic_loss/Select*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub*
out_type0*
_output_shapes
:
�
Ltraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Shape_1Shape(loss/activation_2_loss/logistic_loss/mul*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub*
out_type0*
_output_shapes
:
�
Ztraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/ShapeLtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Shape_1*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub*2
_output_shapes 
:���������:���������
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/SumSumHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/ReshapeZtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub
�
Ltraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/ReshapeReshapeHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/SumJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Shape*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub*
Tshape0*'
_output_shapes
:���������
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Sum_1SumHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Reshape\training/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub*
_output_shapes
:
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/NegNegJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Sum_1*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub*
_output_shapes
:
�
Ntraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Reshape_1ReshapeHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/NegLtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Shape_1*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/sub*
Tshape0*0
_output_shapes
:������������������
�
Ltraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/add/xConstK^training/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Reshape_1*=
_class3
1/loc:@loss/activation_2_loss/logistic_loss/Log1p*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/addAddLtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/add/x(loss/activation_2_loss/logistic_loss/Exp*
T0*=
_class3
1/loc:@loss/activation_2_loss/logistic_loss/Log1p*'
_output_shapes
:���������
�
Qtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/Reciprocal
ReciprocalJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/add*=
_class3
1/loc:@loss/activation_2_loss/logistic_loss/Log1p*'
_output_shapes
:���������*
T0
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/mulMulJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss_grad/Reshape_1Qtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/Reciprocal*'
_output_shapes
:���������*
T0*=
_class3
1/loc:@loss/activation_2_loss/logistic_loss/Log1p
�
Rtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_grad/zeros_like	ZerosLikeloss/activation_2_loss/Log*
T0*>
_class4
20loc:@loss/activation_2_loss/logistic_loss/Select*'
_output_shapes
:���������
�
Ntraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_grad/SelectSelect1loss/activation_2_loss/logistic_loss/GreaterEqualLtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/ReshapeRtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_grad/zeros_like*
T0*>
_class4
20loc:@loss/activation_2_loss/logistic_loss/Select*'
_output_shapes
:���������
�
Ptraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_grad/Select_1Select1loss/activation_2_loss/logistic_loss/GreaterEqualRtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_grad/zeros_likeLtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Reshape*
T0*>
_class4
20loc:@loss/activation_2_loss/logistic_loss/Select*'
_output_shapes
:���������
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/ShapeShapeloss/activation_2_loss/Log*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*
out_type0*
_output_shapes
:
�
Ltraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Shape_1Shapeactivation_2_target*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*
out_type0*
_output_shapes
:
�
Ztraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgsJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/ShapeLtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Shape_1*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*2
_output_shapes 
:���������:���������
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/MulMulNtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Reshape_1activation_2_target*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*0
_output_shapes
:������������������*
T0
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/SumSumHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/MulZtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*
_output_shapes
:
�
Ltraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/ReshapeReshapeHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/SumJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Shape*'
_output_shapes
:���������*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*
Tshape0
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Mul_1Mulloss/activation_2_loss/LogNtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/sub_grad/Reshape_1*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*0
_output_shapes
:������������������
�
Jtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Sum_1SumJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Mul_1\training/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Ntraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Reshape_1ReshapeJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Sum_1Ltraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/Shape_1*0
_output_shapes
:������������������*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/mul*
Tshape0
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Exp_grad/mulMulJtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Log1p_grad/mul(loss/activation_2_loss/logistic_loss/Exp*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/Exp*'
_output_shapes
:���������
�
Ttraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_1_grad/zeros_like	ZerosLike(loss/activation_2_loss/logistic_loss/Neg*
T0*@
_class6
42loc:@loss/activation_2_loss/logistic_loss/Select_1*'
_output_shapes
:���������
�
Ptraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_1_grad/SelectSelect1loss/activation_2_loss/logistic_loss/GreaterEqualHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Exp_grad/mulTtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_1_grad/zeros_like*@
_class6
42loc:@loss/activation_2_loss/logistic_loss/Select_1*'
_output_shapes
:���������*
T0
�
Rtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_1_grad/Select_1Select1loss/activation_2_loss/logistic_loss/GreaterEqualTtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_1_grad/zeros_likeHtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Exp_grad/mul*
T0*@
_class6
42loc:@loss/activation_2_loss/logistic_loss/Select_1*'
_output_shapes
:���������
�
Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Neg_grad/NegNegPtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_1_grad/Select*
T0*;
_class1
/-loc:@loss/activation_2_loss/logistic_loss/Neg*'
_output_shapes
:���������
�
training/SGD/gradients/AddNAddNNtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_grad/SelectLtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/mul_grad/ReshapeRtraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Select_1_grad/Select_1Htraining/SGD/gradients/loss/activation_2_loss/logistic_loss/Neg_grad/Neg*
T0*>
_class4
20loc:@loss/activation_2_loss/logistic_loss/Select*
N*'
_output_shapes
:���������
�
Atraining/SGD/gradients/loss/activation_2_loss/Log_grad/Reciprocal
Reciprocalloss/activation_2_loss/truediv^training/SGD/gradients/AddN*
T0*-
_class#
!loc:@loss/activation_2_loss/Log*'
_output_shapes
:���������
�
:training/SGD/gradients/loss/activation_2_loss/Log_grad/mulMultraining/SGD/gradients/AddNAtraining/SGD/gradients/loss/activation_2_loss/Log_grad/Reciprocal*
T0*-
_class#
!loc:@loss/activation_2_loss/Log*'
_output_shapes
:���������
�
@training/SGD/gradients/loss/activation_2_loss/truediv_grad/ShapeShape$loss/activation_2_loss/clip_by_value*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*
out_type0*
_output_shapes
:
�
Btraining/SGD/gradients/loss/activation_2_loss/truediv_grad/Shape_1Shapeloss/activation_2_loss/sub_1*
_output_shapes
:*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*
out_type0
�
Ptraining/SGD/gradients/loss/activation_2_loss/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs@training/SGD/gradients/loss/activation_2_loss/truediv_grad/ShapeBtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*2
_output_shapes 
:���������:���������
�
Btraining/SGD/gradients/loss/activation_2_loss/truediv_grad/RealDivRealDiv:training/SGD/gradients/loss/activation_2_loss/Log_grad/mulloss/activation_2_loss/sub_1*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*'
_output_shapes
:���������
�
>training/SGD/gradients/loss/activation_2_loss/truediv_grad/SumSumBtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/RealDivPtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv
�
Btraining/SGD/gradients/loss/activation_2_loss/truediv_grad/ReshapeReshape>training/SGD/gradients/loss/activation_2_loss/truediv_grad/Sum@training/SGD/gradients/loss/activation_2_loss/truediv_grad/Shape*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*
Tshape0*'
_output_shapes
:���������
�
>training/SGD/gradients/loss/activation_2_loss/truediv_grad/NegNeg$loss/activation_2_loss/clip_by_value*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*'
_output_shapes
:���������
�
Dtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/RealDiv_1RealDiv>training/SGD/gradients/loss/activation_2_loss/truediv_grad/Negloss/activation_2_loss/sub_1*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*'
_output_shapes
:���������
�
Dtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/RealDiv_2RealDivDtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/RealDiv_1loss/activation_2_loss/sub_1*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv
�
>training/SGD/gradients/loss/activation_2_loss/truediv_grad/mulMul:training/SGD/gradients/loss/activation_2_loss/Log_grad/mulDtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/RealDiv_2*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv
�
@training/SGD/gradients/loss/activation_2_loss/truediv_grad/Sum_1Sum>training/SGD/gradients/loss/activation_2_loss/truediv_grad/mulRtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*
_output_shapes
:
�
Dtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/Reshape_1Reshape@training/SGD/gradients/loss/activation_2_loss/truediv_grad/Sum_1Btraining/SGD/gradients/loss/activation_2_loss/truediv_grad/Shape_1*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*
Tshape0*'
_output_shapes
:���������
�
>training/SGD/gradients/loss/activation_2_loss/sub_1_grad/ShapeConst*/
_class%
#!loc:@loss/activation_2_loss/sub_1*
valueB *
dtype0*
_output_shapes
: 
�
@training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Shape_1Shape$loss/activation_2_loss/clip_by_value*
T0*/
_class%
#!loc:@loss/activation_2_loss/sub_1*
out_type0*
_output_shapes
:
�
Ntraining/SGD/gradients/loss/activation_2_loss/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs>training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Shape@training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Shape_1*
T0*/
_class%
#!loc:@loss/activation_2_loss/sub_1*2
_output_shapes 
:���������:���������
�
<training/SGD/gradients/loss/activation_2_loss/sub_1_grad/SumSumDtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/Reshape_1Ntraining/SGD/gradients/loss/activation_2_loss/sub_1_grad/BroadcastGradientArgs*
T0*/
_class%
#!loc:@loss/activation_2_loss/sub_1*
_output_shapes
:*
	keep_dims( *

Tidx0
�
@training/SGD/gradients/loss/activation_2_loss/sub_1_grad/ReshapeReshape<training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Sum>training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Shape*
T0*/
_class%
#!loc:@loss/activation_2_loss/sub_1*
Tshape0*
_output_shapes
: 
�
>training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Sum_1SumDtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/Reshape_1Ptraining/SGD/gradients/loss/activation_2_loss/sub_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*/
_class%
#!loc:@loss/activation_2_loss/sub_1
�
<training/SGD/gradients/loss/activation_2_loss/sub_1_grad/NegNeg>training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Sum_1*
T0*/
_class%
#!loc:@loss/activation_2_loss/sub_1*
_output_shapes
:
�
Btraining/SGD/gradients/loss/activation_2_loss/sub_1_grad/Reshape_1Reshape<training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Neg@training/SGD/gradients/loss/activation_2_loss/sub_1_grad/Shape_1*
T0*/
_class%
#!loc:@loss/activation_2_loss/sub_1*
Tshape0*'
_output_shapes
:���������
�
training/SGD/gradients/AddN_1AddNBtraining/SGD/gradients/loss/activation_2_loss/truediv_grad/ReshapeBtraining/SGD/gradients/loss/activation_2_loss/sub_1_grad/Reshape_1*
T0*1
_class'
%#loc:@loss/activation_2_loss/truediv*
N*'
_output_shapes
:���������
�
Ftraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/ShapeShapeactivation_2/Sigmoid*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
out_type0*
_output_shapes
:
�
Htraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_1Const*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
valueB *
dtype0*
_output_shapes
: 
�
Htraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_2Const*
_output_shapes
: *7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
valueB *
dtype0
�
Htraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_3Shapetraining/SGD/gradients/AddN_1*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
out_type0*
_output_shapes
:
�
Ltraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/zeros/ConstConst*
_output_shapes
: *7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
valueB
 *    *
dtype0
�
Ftraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/zerosFillHtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_3Ltraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/zeros/Const*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*

index_type0*'
_output_shapes
:���������
�
Etraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/LessLessactivation_2/Sigmoidloss/activation_2_loss/Const*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*'
_output_shapes
:���������
�
Htraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/GreaterGreateractivation_2/Sigmoidloss/activation_2_loss/sub*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value
�
Vtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsFtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/ShapeHtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_1*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*2
_output_shapes 
:���������:���������
�
Xtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/BroadcastGradientArgs_1BroadcastGradientArgsFtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/ShapeHtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_2*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*2
_output_shapes 
:���������:���������
�
Jtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/LogicalOr	LogicalOrEtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/LessHtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Greater*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*'
_output_shapes
:���������
�
Gtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/SelectSelectJtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/LogicalOrFtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/zerostraining/SGD/gradients/AddN_1*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*'
_output_shapes
:���������
�
Itraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Select_1SelectEtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Lesstraining/SGD/gradients/AddN_1Ftraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/zeros*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value
�
Itraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Select_2SelectHtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Greatertraining/SGD/gradients/AddN_1Ftraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/zeros*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*'
_output_shapes
:���������
�
Dtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/SumSumGtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/SelectXtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/BroadcastGradientArgs_1*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
_output_shapes
:*
	keep_dims( *

Tidx0
�
Htraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/ReshapeReshapeDtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/SumFtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape*'
_output_shapes
:���������*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
Tshape0
�
Ftraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Sum_1SumItraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Select_1Xtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value
�
Jtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Reshape_1ReshapeFtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Sum_1Htraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_1*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
Tshape0*
_output_shapes
: 
�
Ftraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Sum_2SumItraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Select_2Ztraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/BroadcastGradientArgs_1:1*
	keep_dims( *

Tidx0*
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
_output_shapes
:
�
Jtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Reshape_2ReshapeFtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Sum_2Htraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Shape_2*
_output_shapes
: *
T0*7
_class-
+)loc:@loss/activation_2_loss/clip_by_value*
Tshape0
�
<training/SGD/gradients/activation_2/Sigmoid_grad/SigmoidGradSigmoidGradactivation_2/SigmoidHtraining/SGD/gradients/loss/activation_2_loss/clip_by_value_grad/Reshape*
T0*'
_class
loc:@activation_2/Sigmoid*'
_output_shapes
:���������
�
7training/SGD/gradients/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad<training/SGD/gradients/activation_2/Sigmoid_grad/SigmoidGrad*
T0*"
_class
loc:@dense_2/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
1training/SGD/gradients/dense_2/MatMul_grad/MatMulMatMul<training/SGD/gradients/activation_2/Sigmoid_grad/SigmoidGraddense_2/kernel/read*
transpose_b(*
T0*!
_class
loc:@dense_2/MatMul*'
_output_shapes
:���������*
transpose_a( 
�
3training/SGD/gradients/dense_2/MatMul_grad/MatMul_1MatMulactivation_1/Tanh<training/SGD/gradients/activation_2/Sigmoid_grad/SigmoidGrad*
_output_shapes

:*
transpose_a(*
transpose_b( *
T0*!
_class
loc:@dense_2/MatMul
�
6training/SGD/gradients/activation_1/Tanh_grad/TanhGradTanhGradactivation_1/Tanh1training/SGD/gradients/dense_2/MatMul_grad/MatMul*
T0*$
_class
loc:@activation_1/Tanh*'
_output_shapes
:���������
�
7training/SGD/gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad6training/SGD/gradients/activation_1/Tanh_grad/TanhGrad*
T0*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
_output_shapes
:
�
1training/SGD/gradients/dense_1/MatMul_grad/MatMulMatMul6training/SGD/gradients/activation_1/Tanh_grad/TanhGraddense_1/kernel/read*
T0*!
_class
loc:@dense_1/MatMul*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
3training/SGD/gradients/dense_1/MatMul_grad/MatMul_1MatMuldense_1_input6training/SGD/gradients/activation_1/Tanh_grad/TanhGrad*
T0*!
_class
loc:@dense_1/MatMul*
_output_shapes

:*
transpose_a(*
transpose_b( 
^
training/SGD/AssignAdd/valueConst*
dtype0	*
_output_shapes
: *
value	B	 R
�
training/SGD/AssignAdd	AssignAddSGD/iterationstraining/SGD/AssignAdd/value*
use_locking( *
T0	*!
_class
loc:@SGD/iterations*
_output_shapes
: 
g
training/SGD/zerosConst*
_output_shapes

:*
valueB*    *
dtype0
�
training/SGD/Variable
VariableV2*
shared_name *
dtype0*
_output_shapes

:*
	container *
shape
:
�
training/SGD/Variable/AssignAssigntraining/SGD/Variabletraining/SGD/zeros*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*
_output_shapes

:*
use_locking(
�
training/SGD/Variable/readIdentitytraining/SGD/Variable*
_output_shapes

:*
T0*(
_class
loc:@training/SGD/Variable
a
training/SGD/zeros_1Const*
_output_shapes
:*
valueB*    *
dtype0
�
training/SGD/Variable_1
VariableV2*
shared_name *
dtype0*
_output_shapes
:*
	container *
shape:
�
training/SGD/Variable_1/AssignAssigntraining/SGD/Variable_1training/SGD/zeros_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_1
�
training/SGD/Variable_1/readIdentitytraining/SGD/Variable_1*
T0**
_class 
loc:@training/SGD/Variable_1*
_output_shapes
:
i
training/SGD/zeros_2Const*
valueB*    *
dtype0*
_output_shapes

:
�
training/SGD/Variable_2
VariableV2*
shape
:*
shared_name *
dtype0*
_output_shapes

:*
	container 
�
training/SGD/Variable_2/AssignAssigntraining/SGD/Variable_2training/SGD/zeros_2**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
training/SGD/Variable_2/readIdentitytraining/SGD/Variable_2*
_output_shapes

:*
T0**
_class 
loc:@training/SGD/Variable_2
a
training/SGD/zeros_3Const*
dtype0*
_output_shapes
:*
valueB*    
�
training/SGD/Variable_3
VariableV2*
dtype0*
_output_shapes
:*
	container *
shape:*
shared_name 
�
training/SGD/Variable_3/AssignAssigntraining/SGD/Variable_3training/SGD/zeros_3*
T0**
_class 
loc:@training/SGD/Variable_3*
validate_shape(*
_output_shapes
:*
use_locking(
�
training/SGD/Variable_3/readIdentitytraining/SGD/Variable_3*
T0**
_class 
loc:@training/SGD/Variable_3*
_output_shapes
:
o
training/SGD/mulMulSGD/momentum/readtraining/SGD/Variable/read*
T0*
_output_shapes

:
�
training/SGD/mul_1MulSGD/lr/read3training/SGD/gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
f
training/SGD/subSubtraining/SGD/multraining/SGD/mul_1*
_output_shapes

:*
T0
�
training/SGD/AssignAssigntraining/SGD/Variabletraining/SGD/sub*(
_class
loc:@training/SGD/Variable*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
g
training/SGD/addAdddense_1/kernel/readtraining/SGD/sub*
T0*
_output_shapes

:
�
training/SGD/Assign_1Assigndense_1/kerneltraining/SGD/add*
use_locking(*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:
o
training/SGD/mul_2MulSGD/momentum/readtraining/SGD/Variable_1/read*
_output_shapes
:*
T0
�
training/SGD/mul_3MulSGD/lr/read7training/SGD/gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
f
training/SGD/sub_1Subtraining/SGD/mul_2training/SGD/mul_3*
T0*
_output_shapes
:
�
training/SGD/Assign_2Assigntraining/SGD/Variable_1training/SGD/sub_1*
T0**
_class 
loc:@training/SGD/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(
e
training/SGD/add_1Adddense_1/bias/readtraining/SGD/sub_1*
_output_shapes
:*
T0
�
training/SGD/Assign_3Assigndense_1/biastraining/SGD/add_1*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
s
training/SGD/mul_4MulSGD/momentum/readtraining/SGD/Variable_2/read*
T0*
_output_shapes

:
�
training/SGD/mul_5MulSGD/lr/read3training/SGD/gradients/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:*
T0
j
training/SGD/sub_2Subtraining/SGD/mul_4training/SGD/mul_5*
T0*
_output_shapes

:
�
training/SGD/Assign_4Assigntraining/SGD/Variable_2training/SGD/sub_2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2
k
training/SGD/add_2Adddense_2/kernel/readtraining/SGD/sub_2*
T0*
_output_shapes

:
�
training/SGD/Assign_5Assigndense_2/kerneltraining/SGD/add_2*
use_locking(*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:
o
training/SGD/mul_6MulSGD/momentum/readtraining/SGD/Variable_3/read*
T0*
_output_shapes
:
�
training/SGD/mul_7MulSGD/lr/read7training/SGD/gradients/dense_2/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
f
training/SGD/sub_3Subtraining/SGD/mul_6training/SGD/mul_7*
T0*
_output_shapes
:
�
training/SGD/Assign_6Assigntraining/SGD/Variable_3training/SGD/sub_3*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_3*
validate_shape(*
_output_shapes
:
e
training/SGD/add_3Adddense_2/bias/readtraining/SGD/sub_3*
T0*
_output_shapes
:
�
training/SGD/Assign_7Assigndense_2/biastraining/SGD/add_3*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(*
_output_shapes
:
�
training/group_depsNoOp	^loss/mul^training/SGD/Assign^training/SGD/AssignAdd^training/SGD/Assign_1^training/SGD/Assign_2^training/SGD/Assign_3^training/SGD/Assign_4^training/SGD/Assign_5^training/SGD/Assign_6^training/SGD/Assign_7
�
IsVariableInitializedIsVariableInitializeddense_1/kernel*
dtype0*
_output_shapes
: *!
_class
loc:@dense_1/kernel
�
IsVariableInitialized_1IsVariableInitializeddense_1/bias*
_class
loc:@dense_1/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_2IsVariableInitializeddense_2/kernel*!
_class
loc:@dense_2/kernel*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_3IsVariableInitializeddense_2/bias*
_class
loc:@dense_2/bias*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_4IsVariableInitializedSGD/iterations*!
_class
loc:@SGD/iterations*
dtype0	*
_output_shapes
: 
x
IsVariableInitialized_5IsVariableInitializedSGD/lr*
_class
loc:@SGD/lr*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_6IsVariableInitializedSGD/momentum*
_class
loc:@SGD/momentum*
dtype0*
_output_shapes
: 
~
IsVariableInitialized_7IsVariableInitialized	SGD/decay*
_class
loc:@SGD/decay*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_8IsVariableInitializedtraining/SGD/Variable*(
_class
loc:@training/SGD/Variable*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_9IsVariableInitializedtraining/SGD/Variable_1**
_class 
loc:@training/SGD/Variable_1*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_10IsVariableInitializedtraining/SGD/Variable_2**
_class 
loc:@training/SGD/Variable_2*
dtype0*
_output_shapes
: 
�
IsVariableInitialized_11IsVariableInitializedtraining/SGD/Variable_3**
_class 
loc:@training/SGD/Variable_3*
dtype0*
_output_shapes
: 
�
initNoOp^SGD/decay/Assign^SGD/iterations/Assign^SGD/lr/Assign^SGD/momentum/Assign^dense_1/bias/Assign^dense_1/kernel/Assign^dense_2/bias/Assign^dense_2/kernel/Assign^training/SGD/Variable/Assign^training/SGD/Variable_1/Assign^training/SGD/Variable_2/Assign^training/SGD/Variable_3/Assign
)

group_depsNoOp^activation_2/Sigmoid

init_all_tablesNoOp
(
legacy_init_opNoOp^init_all_tables
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_163c56c933974c88bacee2fda37e10b3/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
�
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*�
value�B�B	SGD/decayBSGD/iterationsBSGD/lrBSGD/momentumBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBtraining/SGD/VariableBtraining/SGD/Variable_1Btraining/SGD/Variable_2Btraining/SGD/Variable_3*
dtype0
�
save/SaveV2/shape_and_slicesConst"/device:CPU:0*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slices	SGD/decaySGD/iterationsSGD/lrSGD/momentumdense_1/biasdense_1/kerneldense_2/biasdense_2/kerneltraining/SGD/Variabletraining/SGD/Variable_1training/SGD/Variable_2training/SGD/Variable_3"/device:CPU:0*
dtypes
2	
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
_output_shapes
: *
T0*'
_class
loc:@save/ShardedFilename
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*
_output_shapes
:*
T0*

axis 
�
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(
�
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency"/device:CPU:0*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�B	SGD/decayBSGD/iterationsBSGD/lrBSGD/momentumBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBtraining/SGD/VariableBtraining/SGD/Variable_1Btraining/SGD/Variable_2Btraining/SGD/Variable_3*
dtype0*
_output_shapes
:
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*+
value"B B B B B B B B B B B B B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
2	
�
save/AssignAssign	SGD/decaysave/RestoreV2*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@SGD/decay*
validate_shape(
�
save/Assign_1AssignSGD/iterationssave/RestoreV2:1*
use_locking(*
T0	*!
_class
loc:@SGD/iterations*
validate_shape(*
_output_shapes
: 
�
save/Assign_2AssignSGD/lrsave/RestoreV2:2*
use_locking(*
T0*
_class
loc:@SGD/lr*
validate_shape(*
_output_shapes
: 
�
save/Assign_3AssignSGD/momentumsave/RestoreV2:3*
use_locking(*
T0*
_class
loc:@SGD/momentum*
validate_shape(*
_output_shapes
: 
�
save/Assign_4Assigndense_1/biassave/RestoreV2:4*
use_locking(*
T0*
_class
loc:@dense_1/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_5Assigndense_1/kernelsave/RestoreV2:5*
T0*!
_class
loc:@dense_1/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_6Assigndense_2/biassave/RestoreV2:6*
_output_shapes
:*
use_locking(*
T0*
_class
loc:@dense_2/bias*
validate_shape(
�
save/Assign_7Assigndense_2/kernelsave/RestoreV2:7*
T0*!
_class
loc:@dense_2/kernel*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_8Assigntraining/SGD/Variablesave/RestoreV2:8*
T0*(
_class
loc:@training/SGD/Variable*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_9Assigntraining/SGD/Variable_1save/RestoreV2:9**
_class 
loc:@training/SGD/Variable_1*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_10Assigntraining/SGD/Variable_2save/RestoreV2:10*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_2*
validate_shape(*
_output_shapes

:
�
save/Assign_11Assigntraining/SGD/Variable_3save/RestoreV2:11*
use_locking(*
T0**
_class 
loc:@training/SGD/Variable_3*
validate_shape(*
_output_shapes
:
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard"<
save/Const:0save/Identity:0save/restore_all (5 @F8"�
	variables��
Z
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:0
K
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:0
Z
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:0
K
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:0
`
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:0
@
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:0
X
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:0
L
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:0
k
training/SGD/Variable:0training/SGD/Variable/Assigntraining/SGD/Variable/read:02training/SGD/zeros:0
s
training/SGD/Variable_1:0training/SGD/Variable_1/Assigntraining/SGD/Variable_1/read:02training/SGD/zeros_1:0
s
training/SGD/Variable_2:0training/SGD/Variable_2/Assigntraining/SGD/Variable_2/read:02training/SGD/zeros_2:0
s
training/SGD/Variable_3:0training/SGD/Variable_3/Assigntraining/SGD/Variable_3/read:02training/SGD/zeros_3:0"$
legacy_init_op

legacy_init_op"�	
trainable_variables��
Z
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02dense_1/random_uniform:0
K
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02dense_1/Const:0
Z
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02dense_2/random_uniform:0
K
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02dense_2/Const:0
`
SGD/iterations:0SGD/iterations/AssignSGD/iterations/read:02SGD/iterations/initial_value:0
@
SGD/lr:0SGD/lr/AssignSGD/lr/read:02SGD/lr/initial_value:0
X
SGD/momentum:0SGD/momentum/AssignSGD/momentum/read:02SGD/momentum/initial_value:0
L
SGD/decay:0SGD/decay/AssignSGD/decay/read:02SGD/decay/initial_value:0
k
training/SGD/Variable:0training/SGD/Variable/Assigntraining/SGD/Variable/read:02training/SGD/zeros:0
s
training/SGD/Variable_1:0training/SGD/Variable_1/Assigntraining/SGD/Variable_1/read:02training/SGD/zeros_1:0
s
training/SGD/Variable_2:0training/SGD/Variable_2/Assigntraining/SGD/Variable_2/read:02training/SGD/zeros_2:0
s
training/SGD/Variable_3:0training/SGD/Variable_3/Assigntraining/SGD/Variable_3/read:02training/SGD/zeros_3:0*�
serving_default�
0
inputs&
dense_1_input:0���������;

prediction-
activation_2/Sigmoid:0���������tensorflow/serving/predict