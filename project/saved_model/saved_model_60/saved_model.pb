??5
?#?#
D
AddV2
x"T
y"T
z"T"
Ttype:
2	??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
Z
BroadcastTo

input"T
shape"Tidx
output"T"	
Ttype"
Tidxtype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Expm1
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
?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
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
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
?
RandomUniformInt

shape"T
minval"Tout
maxval"Tout
output"Tout"
seedint "
seed2int "
Touttype:
2	"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
@
Softplus
features"T
activations"T"
Ttype:
2
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	?
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
^
StatelessRandomGetKeyCounter
seed"Tseed
key
counter"
Tseedtype0	:
2	
?
StatelessRandomUniformIntV2
shape"Tshape
key
counter
alg
minval"dtype
maxval"dtype
output"dtype"
dtypetype:
2	"
Tshapetype0:
2	
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.8.0-rc12v2.8.0-rc0-49-g244b9d77fd48??3
?
#conv2d_flipout/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#conv2d_flipout/kernel_posterior_loc
?
7conv2d_flipout/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp#conv2d_flipout/kernel_posterior_loc*&
_output_shapes
: *
dtype0
?
3conv2d_flipout/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: *D
shared_name53conv2d_flipout/kernel_posterior_untransformed_scale
?
Gconv2d_flipout/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp3conv2d_flipout/kernel_posterior_untransformed_scale*&
_output_shapes
: *
dtype0
?
!conv2d_flipout/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!conv2d_flipout/bias_posterior_loc
?
5conv2d_flipout/bias_posterior_loc/Read/ReadVariableOpReadVariableOp!conv2d_flipout/bias_posterior_loc*
_output_shapes
: *
dtype0
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
?
%conv2d_flipout_1/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*6
shared_name'%conv2d_flipout_1/kernel_posterior_loc
?
9conv2d_flipout_1/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp%conv2d_flipout_1/kernel_posterior_loc*&
_output_shapes
: @*
dtype0
?
5conv2d_flipout_1/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*F
shared_name75conv2d_flipout_1/kernel_posterior_untransformed_scale
?
Iconv2d_flipout_1/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp5conv2d_flipout_1/kernel_posterior_untransformed_scale*&
_output_shapes
: @*
dtype0
?
#conv2d_flipout_1/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#conv2d_flipout_1/bias_posterior_loc
?
7conv2d_flipout_1/bias_posterior_loc/Read/ReadVariableOpReadVariableOp#conv2d_flipout_1/bias_posterior_loc*
_output_shapes
:@*
dtype0
?
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_1/gamma
?
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
:@*
dtype0
?
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_1/beta
?
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
:@*
dtype0
?
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_1/moving_mean
?
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
:@*
dtype0
?
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_1/moving_variance
?
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
:@*
dtype0
?
"dense_flipout/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*3
shared_name$"dense_flipout/kernel_posterior_loc
?
6dense_flipout/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp"dense_flipout/kernel_posterior_loc* 
_output_shapes
:
??*
dtype0
?
2dense_flipout/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*C
shared_name42dense_flipout/kernel_posterior_untransformed_scale
?
Fdense_flipout/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp2dense_flipout/kernel_posterior_untransformed_scale* 
_output_shapes
:
??*
dtype0
?
 dense_flipout/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*1
shared_name" dense_flipout/bias_posterior_loc
?
4dense_flipout/bias_posterior_loc/Read/ReadVariableOpReadVariableOp dense_flipout/bias_posterior_loc*
_output_shapes	
:?*
dtype0
?
$dense_flipout_1/kernel_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*5
shared_name&$dense_flipout_1/kernel_posterior_loc
?
8dense_flipout_1/kernel_posterior_loc/Read/ReadVariableOpReadVariableOp$dense_flipout_1/kernel_posterior_loc*
_output_shapes
:	?
*
dtype0
?
4dense_flipout_1/kernel_posterior_untransformed_scaleVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*E
shared_name64dense_flipout_1/kernel_posterior_untransformed_scale
?
Hdense_flipout_1/kernel_posterior_untransformed_scale/Read/ReadVariableOpReadVariableOp4dense_flipout_1/kernel_posterior_untransformed_scale*
_output_shapes
:	?
*
dtype0
?
"dense_flipout_1/bias_posterior_locVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*3
shared_name$"dense_flipout_1/bias_posterior_loc
?
6dense_flipout_1/bias_posterior_loc/Read/ReadVariableOpReadVariableOp"dense_flipout_1/bias_posterior_loc*
_output_shapes
:
*
dtype0
J
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
l
Const_1Const*&
_output_shapes
: *
dtype0*%
valueB *    
L
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
l
Const_3Const*&
_output_shapes
: @*
dtype0*%
valueB @*    
L
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
`
Const_5Const* 
_output_shapes
:
??*
dtype0*
valueB
??*    
L
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
^
Const_7Const*
_output_shapes
:	?
*
dtype0*
valueB	?
*    

NoOpNoOp
?S
Const_8Const"/device:CPU:0*
_output_shapes
: *
dtype0*?S
value?SB?S B?R
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
?
kernel_posterior_loc
($kernel_posterior_untransformed_scale
kernel_posterior
kernel_prior
bias_posterior_loc
bias_posterior
kernel_posterior_affine
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
?
axis
	 gamma
!beta
"moving_mean
#moving_variance
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses*
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
?
0kernel_posterior_loc
(1$kernel_posterior_untransformed_scale
2kernel_posterior
3kernel_prior
4bias_posterior_loc
5bias_posterior
6kernel_posterior_affine
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses*
?
=axis
	>gamma
?beta
@moving_mean
Amoving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses*
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses* 
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses* 
?
Tkernel_posterior_loc
(U$kernel_posterior_untransformed_scale
Vkernel_posterior
Wkernel_prior
Xbias_posterior_loc
Ybias_posterior
Zkernel_posterior_affine
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
?
akernel_posterior_loc
(b$kernel_posterior_untransformed_scale
ckernel_posterior
dkernel_prior
ebias_posterior_loc
fbias_posterior
gkernel_posterior_affine
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*
?
0
1
2
 3
!4
"5
#6
07
18
49
>10
?11
@12
A13
T14
U15
X16
a17
b18
e19*
z
0
1
2
 3
!4
05
16
47
>8
?9
T10
U11
X12
a13
b14
e15*
* 
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

sserving_default* 
?{
VARIABLE_VALUE#conv2d_flipout/kernel_posterior_locDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE3conv2d_flipout/kernel_posterior_untransformed_scaleTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
+
t_distribution
u_graph_parents*
)
v_distribution
w_graph_parents* 
}w
VARIABLE_VALUE!conv2d_flipout/bias_posterior_locBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
+
x_distribution
y_graph_parents*
$

z_scale
{_graph_parents*

0
1
2*

0
1
2*
* 
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
hb
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
tn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
 0
!1
"2
#3*

 0
!1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 
* 
* 
?}
VARIABLE_VALUE%conv2d_flipout_1/kernel_posterior_locDlayer_with_weights-2/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE5conv2d_flipout_1/kernel_posterior_untransformed_scaleTlayer_with_weights-2/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
-
?_distribution
?_graph_parents*
+
?_distribution
?_graph_parents* 
y
VARIABLE_VALUE#conv2d_flipout_1/bias_posterior_locBlayer_with_weights-2/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
-
?_distribution
?_graph_parents*
&
?_scale
?_graph_parents*

00
11
42*

00
11
42*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
>0
?1
@2
A3*

>0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses* 
* 
* 
?z
VARIABLE_VALUE"dense_flipout/kernel_posterior_locDlayer_with_weights-4/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE2dense_flipout/kernel_posterior_untransformed_scaleTlayer_with_weights-4/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
-
?_distribution
?_graph_parents*
+
?_distribution
?_graph_parents* 
|v
VARIABLE_VALUE dense_flipout/bias_posterior_locBlayer_with_weights-4/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
-
?_distribution
?_graph_parents*
&
?_scale
?_graph_parents*

T0
U1
X2*

T0
U1
X2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
?|
VARIABLE_VALUE$dense_flipout_1/kernel_posterior_locDlayer_with_weights-5/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE4dense_flipout_1/kernel_posterior_untransformed_scaleTlayer_with_weights-5/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUE*
-
?_distribution
?_graph_parents*
+
?_distribution
?_graph_parents* 
~x
VARIABLE_VALUE"dense_flipout_1/bias_posterior_locBlayer_with_weights-5/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUE*
-
?_distribution
?_graph_parents*
&
?_scale
?_graph_parents*

a0
b1
e2*

a0
b1
e2*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*
* 
* 
 
"0
#1
@2
A3*
C
0
1
2
3
4
5
6
7
	8*
* 
* 
* 
* 
/
_loc

z_scale
?_graph_parents*
* 

?_graph_parents* 
* 
#
_loc
?_graph_parents*
* 

_pretransformed_input*
* 
* 
* 
* 
* 
* 

"0
#1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
0
0_loc
?_scale
?_graph_parents*
* 

?_graph_parents* 
* 
#
4_loc
?_graph_parents*
* 

1_pretransformed_input*
* 
* 
* 
* 
* 
* 

@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
0
T_loc
?_scale
?_graph_parents*
* 

?_graph_parents* 
* 
#
X_loc
?_graph_parents*
* 

U_pretransformed_input*
* 
* 
* 
* 
* 
* 
0
a_loc
?_scale
?_graph_parents*
* 

?_graph_parents* 
* 
#
e_loc
?_graph_parents*
* 

b_pretransformed_input*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
?
$serving_default_conv2d_flipout_inputPlaceholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?	
StatefulPartitionedCallStatefulPartitionedCall$serving_default_conv2d_flipout_input3conv2d_flipout/kernel_posterior_untransformed_scale#conv2d_flipout/kernel_posterior_loc!conv2d_flipout/bias_posterior_locConstConst_1batch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance5conv2d_flipout_1/kernel_posterior_untransformed_scale%conv2d_flipout_1/kernel_posterior_loc#conv2d_flipout_1/bias_posterior_locConst_2Const_3batch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance2dense_flipout/kernel_posterior_untransformed_scale"dense_flipout/kernel_posterior_loc dense_flipout/bias_posterior_locConst_4Const_54dense_flipout_1/kernel_posterior_untransformed_scale$dense_flipout_1/kernel_posterior_loc"dense_flipout_1/bias_posterior_locConst_6Const_7*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2559734
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename7conv2d_flipout/kernel_posterior_loc/Read/ReadVariableOpGconv2d_flipout/kernel_posterior_untransformed_scale/Read/ReadVariableOp5conv2d_flipout/bias_posterior_loc/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp9conv2d_flipout_1/kernel_posterior_loc/Read/ReadVariableOpIconv2d_flipout_1/kernel_posterior_untransformed_scale/Read/ReadVariableOp7conv2d_flipout_1/bias_posterior_loc/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp6dense_flipout/kernel_posterior_loc/Read/ReadVariableOpFdense_flipout/kernel_posterior_untransformed_scale/Read/ReadVariableOp4dense_flipout/bias_posterior_loc/Read/ReadVariableOp8dense_flipout_1/kernel_posterior_loc/Read/ReadVariableOpHdense_flipout_1/kernel_posterior_untransformed_scale/Read/ReadVariableOp6dense_flipout_1/bias_posterior_loc/Read/ReadVariableOpConst_8*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_save_2560635
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename#conv2d_flipout/kernel_posterior_loc3conv2d_flipout/kernel_posterior_untransformed_scale!conv2d_flipout/bias_posterior_locbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance%conv2d_flipout_1/kernel_posterior_loc5conv2d_flipout_1/kernel_posterior_untransformed_scale#conv2d_flipout_1/bias_posterior_locbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_variance"dense_flipout/kernel_posterior_loc2dense_flipout/kernel_posterior_untransformed_scale dense_flipout/bias_posterior_loc$dense_flipout_1/kernel_posterior_loc4dense_flipout_1/kernel_posterior_untransformed_scale"dense_flipout_1/bias_posterior_loc* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__traced_restore_2560705??2
??
?
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2560150

inputsH
.normal_sample_softplus_readvariableop_resource: @8
conv2d_readvariableop_resource: @z
lindependentdeterministic_constructed_at_conv2d_flipout_1_sample_deterministic_sample_readvariableop_resource:@?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560122?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??Conv2D/ReadVariableOp?cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOps
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*&
_output_shapes
: @]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: @v
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskx
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"          @   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0**
_output_shapes
: @*
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: @?
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: @?
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0**
_output_shapes
: @}
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0**
_output_shapes
: @t
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: @|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsstrided_slice_1:output:0ExpandDims_1/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2ExpandDims:output:0ExpandDims_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:u
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? T
ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B :@R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : w
ExpandDims_2
ExpandDimsExpandDims_2/input:output:0ExpandDims_2/dim:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2ExpandDims:output:0ExpandDims_2:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat_1:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????@*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????@T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????@p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????@R
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_3
ExpandDimsrademacher/Cast:y:0ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:????????? R
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_4
ExpandDimsrademacher_1/Cast:y:0ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:?????????@R
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_5
ExpandDimsExpandDims_3:output:0ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:????????? R
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_6
ExpandDimsExpandDims_4:output:0ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:?????????@c
mulMulinputsExpandDims_5:output:0*
T0*/
_output_shapes
:????????? ?
Conv2D_1Conv2Dmul:z:0Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
p
mul_1MulConv2D_1:output:0ExpandDims_6:output:0*
T0*/
_output_shapes
:?????????@b
addAddV2Conv2D:output:0	mul_1:z:0*
T0*/
_output_shapes
:?????????@?
LIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
rIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOpReadVariableOplindependentdeterministic_constructed_at_conv2d_flipout_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:@*
dtype0?
dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:@?
ZIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
hIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
jIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
jIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_sliceStridedSlicemIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/shape_as_tensor:output:0qIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack:output:0sIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_1:output:0sIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
eIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
gIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgspIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concatConcatV2mIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_0:output:0gIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs:r0:0mIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_2:output:0iIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastToBroadcastTokIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp:value:0dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:@?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
\IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReshapeReshapeiIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastTo:output:0kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:@?
MIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:@?
GIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/ReshapeReshapeeIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape:output:0VIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
:@?
BiasAddBiasAddadd:z:0PIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560122*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560122*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560122*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
{KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^Conv2D/ReadVariableOpd^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:????????? : : : : : @2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOpcIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: @
?
?
,__inference_sequential_layer_call_fn_2558449

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: #
	unknown_8: @#
	unknown_9: @

unknown_10:@

unknown_11

unknown_12

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
??

unknown_18:
??

unknown_19:	?

unknown_20

unknown_21

unknown_22:	?


unknown_23:	?


unknown_24:


unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????
: : : : *2
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2558037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2560194

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
e
I__inference_activation_1_layer_call_and_return_conditional_losses_2557424

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_2558165
conv2d_flipout_input!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: #
	unknown_8: @#
	unknown_9: @

unknown_10:@

unknown_11

unknown_12

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
??

unknown_18:
??

unknown_19:	?

unknown_20

unknown_21

unknown_22:	?


unknown_23:	?


unknown_24:


unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_flipout_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????
: : : : *2
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2558037o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????
.
_user_specified_nameconv2d_flipout_input:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2560212

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
H
,__inference_activation_layer_call_fn_2559972

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2557238h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
/__inference_dense_flipout_layer_call_fn_2560249

inputs
unknown:
??
	unknown_0:
??
	unknown_1:	?
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2557574p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : :
??22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :&"
 
_output_shapes
:
??
??	
?5
G__inference_sequential_layer_call_and_return_conditional_losses_2559671

inputsW
=conv2d_flipout_normal_sample_softplus_readvariableop_resource: G
-conv2d_flipout_conv2d_readvariableop_resource: i
[conv2d_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource: ?
?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559187?
?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: Y
?conv2d_flipout_1_normal_sample_softplus_readvariableop_resource: @I
/conv2d_flipout_1_conv2d_readvariableop_resource: @k
]conv2d_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource:@?
?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559354?
?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@P
<dense_flipout_normal_sample_softplus_readvariableop_resource:
??B
.dense_flipout_matmul_1_readvariableop_resource:
??i
Zdense_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource:	??
?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559506?
?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xQ
>dense_flipout_1_normal_sample_softplus_readvariableop_resource:	?
C
0dense_flipout_1_matmul_1_readvariableop_resource:	?
j
\dense_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource:
?
?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559640?
?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2

identity_3

identity_4??"batch_normalization/AssignNewValue?$batch_normalization/AssignNewValue_1?3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?$batch_normalization_1/AssignNewValue?&batch_normalization_1/AssignNewValue_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$conv2d_flipout/Conv2D/ReadVariableOp?Rconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?4conv2d_flipout/Normal/sample/Softplus/ReadVariableOp?&conv2d_flipout_1/Conv2D/ReadVariableOp?Tconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp?Qdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?%dense_flipout/MatMul_1/ReadVariableOp?3dense_flipout/Normal/sample/Softplus/ReadVariableOp?Sdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?'dense_flipout_1/MatMul_1/ReadVariableOp?5dense_flipout_1/Normal/sample/Softplus/ReadVariableOp~
conv2d_flipout/zeros_likeConst*&
_output_shapes
: *
dtype0*%
valueB *    l
)conv2d_flipout/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
4conv2d_flipout/Normal/sample/Softplus/ReadVariableOpReadVariableOp=conv2d_flipout_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
%conv2d_flipout/Normal/sample/SoftplusSoftplus<conv2d_flipout/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: g
"conv2d_flipout/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
 conv2d_flipout/Normal/sample/addAddV2+conv2d_flipout/Normal/sample/add/x:output:03conv2d_flipout/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: ?
,conv2d_flipout/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             d
"conv2d_flipout/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : z
0conv2d_flipout/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_flipout/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_flipout/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_flipout/Normal/sample/strided_sliceStridedSlice5conv2d_flipout/Normal/sample/shape_as_tensor:output:09conv2d_flipout/Normal/sample/strided_slice/stack:output:0;conv2d_flipout/Normal/sample/strided_slice/stack_1:output:0;conv2d_flipout/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
.conv2d_flipout/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             f
$conv2d_flipout/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : |
2conv2d_flipout/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4conv2d_flipout/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4conv2d_flipout/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,conv2d_flipout/Normal/sample/strided_slice_1StridedSlice7conv2d_flipout/Normal/sample/shape_as_tensor_1:output:0;conv2d_flipout/Normal/sample/strided_slice_1/stack:output:0=conv2d_flipout/Normal/sample/strided_slice_1/stack_1:output:0=conv2d_flipout/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
-conv2d_flipout/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB r
/conv2d_flipout/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
*conv2d_flipout/Normal/sample/BroadcastArgsBroadcastArgs8conv2d_flipout/Normal/sample/BroadcastArgs/s0_1:output:03conv2d_flipout/Normal/sample/strided_slice:output:0*
_output_shapes
:?
,conv2d_flipout/Normal/sample/BroadcastArgs_1BroadcastArgs/conv2d_flipout/Normal/sample/BroadcastArgs:r0:05conv2d_flipout/Normal/sample/strided_slice_1:output:0*
_output_shapes
:v
,conv2d_flipout/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:j
(conv2d_flipout/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#conv2d_flipout/Normal/sample/concatConcatV25conv2d_flipout/Normal/sample/concat/values_0:output:01conv2d_flipout/Normal/sample/BroadcastArgs_1:r0:01conv2d_flipout/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:{
6conv2d_flipout/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    }
8conv2d_flipout/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fconv2d_flipout/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_flipout/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0?
5conv2d_flipout/Normal/sample/normal/random_normal/mulMulOconv2d_flipout/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Aconv2d_flipout/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: ?
1conv2d_flipout/Normal/sample/normal/random_normalAddV29conv2d_flipout/Normal/sample/normal/random_normal/mul:z:0?conv2d_flipout/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: ?
 conv2d_flipout/Normal/sample/mulMul5conv2d_flipout/Normal/sample/normal/random_normal:z:0$conv2d_flipout/Normal/sample/add:z:0*
T0**
_output_shapes
: ?
"conv2d_flipout/Normal/sample/add_1AddV2$conv2d_flipout/Normal/sample/mul:z:0"conv2d_flipout/zeros_like:output:0*
T0**
_output_shapes
: ?
*conv2d_flipout/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
$conv2d_flipout/Normal/sample/ReshapeReshape&conv2d_flipout/Normal/sample/add_1:z:03conv2d_flipout/Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: ?
$conv2d_flipout/Conv2D/ReadVariableOpReadVariableOp-conv2d_flipout_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_flipout/Conv2DConv2Dinputs,conv2d_flipout/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
J
conv2d_flipout/ShapeShapeinputs*
T0*
_output_shapes
:l
"conv2d_flipout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$conv2d_flipout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$conv2d_flipout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_flipout/strided_sliceStridedSliceconv2d_flipout/Shape:output:0+conv2d_flipout/strided_slice/stack:output:0-conv2d_flipout/strided_slice/stack_1:output:0-conv2d_flipout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
conv2d_flipout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/ExpandDims
ExpandDims%conv2d_flipout/strided_slice:output:0&conv2d_flipout/ExpandDims/dim:output:0*
T0*
_output_shapes
:w
$conv2d_flipout/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????p
&conv2d_flipout/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_flipout/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_flipout/strided_slice_1StridedSliceconv2d_flipout/Shape:output:0-conv2d_flipout/strided_slice_1/stack:output:0/conv2d_flipout/strided_slice_1/stack_1:output:0/conv2d_flipout/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
conv2d_flipout/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/ExpandDims_1
ExpandDims'conv2d_flipout/strided_slice_1:output:0(conv2d_flipout/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:\
conv2d_flipout/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/concatConcatV2"conv2d_flipout/ExpandDims:output:0$conv2d_flipout/ExpandDims_1:output:0#conv2d_flipout/concat/axis:output:0*
N*
T0*
_output_shapes
:?
:conv2d_flipout/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8conv2d_flipout/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????~
8conv2d_flipout/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
4conv2d_flipout/rademacher/uniform/sanitize_seed/seedRandomUniformIntCconv2d_flipout/rademacher/uniform/sanitize_seed/seed/shape:output:0Aconv2d_flipout/rademacher/uniform/sanitize_seed/seed/min:output:0Aconv2d_flipout/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
>conv2d_flipout/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
>conv2d_flipout/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Wconv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter=conv2d_flipout/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
>conv2d_flipout/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
:conv2d_flipout/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2conv2d_flipout/concat:output:0]conv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0aconv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Gconv2d_flipout/rademacher/uniform/stateless_random_uniform/alg:output:0Gconv2d_flipout/rademacher/uniform/stateless_random_uniform/min:output:0Gconv2d_flipout/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	a
conv2d_flipout/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher/mulMul(conv2d_flipout/rademacher/mul/x:output:0Cconv2d_flipout/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????a
conv2d_flipout/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher/subSub!conv2d_flipout/rademacher/mul:z:0(conv2d_flipout/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:??????????
conv2d_flipout/rademacher/CastCast!conv2d_flipout/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????c
!conv2d_flipout/ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B : a
conv2d_flipout/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/ExpandDims_2
ExpandDims*conv2d_flipout/ExpandDims_2/input:output:0(conv2d_flipout/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:^
conv2d_flipout/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/concat_1ConcatV2"conv2d_flipout/ExpandDims:output:0$conv2d_flipout/ExpandDims_2:output:0%conv2d_flipout/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
<conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
:conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
6conv2d_flipout/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntEconv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Cconv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/min:output:0Cconv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
@conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
@conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Yconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter?conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
@conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
<conv2d_flipout/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2 conv2d_flipout/concat_1:output:0_conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0cconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Iconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/alg:output:0Iconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/min:output:0Iconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	c
!conv2d_flipout/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher_1/mulMul*conv2d_flipout/rademacher_1/mul/x:output:0Econv2d_flipout/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? c
!conv2d_flipout/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher_1/subSub#conv2d_flipout/rademacher_1/mul:z:0*conv2d_flipout/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? ?
 conv2d_flipout/rademacher_1/CastCast#conv2d_flipout/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? a
conv2d_flipout/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_3
ExpandDims"conv2d_flipout/rademacher/Cast:y:0(conv2d_flipout/ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:?????????a
conv2d_flipout/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_4
ExpandDims$conv2d_flipout/rademacher_1/Cast:y:0(conv2d_flipout/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:????????? a
conv2d_flipout/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_5
ExpandDims$conv2d_flipout/ExpandDims_3:output:0(conv2d_flipout/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????a
conv2d_flipout/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_6
ExpandDims$conv2d_flipout/ExpandDims_4:output:0(conv2d_flipout/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_flipout/mulMulinputs$conv2d_flipout/ExpandDims_5:output:0*
T0*/
_output_shapes
:??????????
conv2d_flipout/Conv2D_1Conv2Dconv2d_flipout/mul:z:0-conv2d_flipout/Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
conv2d_flipout/mul_1Mul conv2d_flipout/Conv2D_1:output:0$conv2d_flipout/ExpandDims_6:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_flipout/addAddV2conv2d_flipout/Conv2D:output:0conv2d_flipout/mul_1:z:0*
T0*/
_output_shapes
:????????? ~
;conv2d_flipout/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Pconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
aconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Rconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOp[conv2d_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0?
Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: ?
Iconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Wconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Yconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Yconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Qconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice\conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0`conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0bconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0bconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Vconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Qconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs_conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Zconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Oconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2\conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Vconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0\conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Xconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Oconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToZconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: ?
Qconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeXconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Zconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ?
<conv2d_flipout/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
6conv2d_flipout/IndependentDeterministic/sample/ReshapeReshapeTconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Econv2d_flipout/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
: ?
conv2d_flipout/BiasAddBiasAddconv2d_flipout/add:z:0?conv2d_flipout/IndependentDeterministic/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp=conv2d_flipout_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559187*
T0*
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp-conv2d_flipout_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559187*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559187*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
 conv2d_flipout/divergence_kernelIdentity?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: ?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_flipout/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0{
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? ?
+conv2d_flipout_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   f
!conv2d_flipout_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
conv2d_flipout_1/zeros_likeFill4conv2d_flipout_1/zeros_like/shape_as_tensor:output:0*conv2d_flipout_1/zeros_like/Const:output:0*
T0*&
_output_shapes
: @n
+conv2d_flipout_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp?conv2d_flipout_1_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
'conv2d_flipout_1/Normal/sample/SoftplusSoftplus>conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @i
$conv2d_flipout_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
"conv2d_flipout_1/Normal/sample/addAddV2-conv2d_flipout_1/Normal/sample/add/x:output:05conv2d_flipout_1/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: @?
.conv2d_flipout_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   f
$conv2d_flipout_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : |
2conv2d_flipout_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4conv2d_flipout_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4conv2d_flipout_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,conv2d_flipout_1/Normal/sample/strided_sliceStridedSlice7conv2d_flipout_1/Normal/sample/shape_as_tensor:output:0;conv2d_flipout_1/Normal/sample/strided_slice/stack:output:0=conv2d_flipout_1/Normal/sample/strided_slice/stack_1:output:0=conv2d_flipout_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
0conv2d_flipout_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"          @   h
&conv2d_flipout_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ~
4conv2d_flipout_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6conv2d_flipout_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6conv2d_flipout_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.conv2d_flipout_1/Normal/sample/strided_slice_1StridedSlice9conv2d_flipout_1/Normal/sample/shape_as_tensor_1:output:0=conv2d_flipout_1/Normal/sample/strided_slice_1/stack:output:0?conv2d_flipout_1/Normal/sample/strided_slice_1/stack_1:output:0?conv2d_flipout_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskr
/conv2d_flipout_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB t
1conv2d_flipout_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
,conv2d_flipout_1/Normal/sample/BroadcastArgsBroadcastArgs:conv2d_flipout_1/Normal/sample/BroadcastArgs/s0_1:output:05conv2d_flipout_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
.conv2d_flipout_1/Normal/sample/BroadcastArgs_1BroadcastArgs1conv2d_flipout_1/Normal/sample/BroadcastArgs:r0:07conv2d_flipout_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:x
.conv2d_flipout_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:l
*conv2d_flipout_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%conv2d_flipout_1/Normal/sample/concatConcatV27conv2d_flipout_1/Normal/sample/concat/values_0:output:03conv2d_flipout_1/Normal/sample/BroadcastArgs_1:r0:03conv2d_flipout_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:}
8conv2d_flipout_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
:conv2d_flipout_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Hconv2d_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal.conv2d_flipout_1/Normal/sample/concat:output:0*
T0**
_output_shapes
: @*
dtype0?
7conv2d_flipout_1/Normal/sample/normal/random_normal/mulMulQconv2d_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Cconv2d_flipout_1/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: @?
3conv2d_flipout_1/Normal/sample/normal/random_normalAddV2;conv2d_flipout_1/Normal/sample/normal/random_normal/mul:z:0Aconv2d_flipout_1/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: @?
"conv2d_flipout_1/Normal/sample/mulMul7conv2d_flipout_1/Normal/sample/normal/random_normal:z:0&conv2d_flipout_1/Normal/sample/add:z:0*
T0**
_output_shapes
: @?
$conv2d_flipout_1/Normal/sample/add_1AddV2&conv2d_flipout_1/Normal/sample/mul:z:0$conv2d_flipout_1/zeros_like:output:0*
T0**
_output_shapes
: @?
,conv2d_flipout_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   ?
&conv2d_flipout_1/Normal/sample/ReshapeReshape(conv2d_flipout_1/Normal/sample/add_1:z:05conv2d_flipout_1/Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: @?
&conv2d_flipout_1/Conv2D/ReadVariableOpReadVariableOp/conv2d_flipout_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_flipout_1/Conv2DConv2Dactivation/Relu:activations:0.conv2d_flipout_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
c
conv2d_flipout_1/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_flipout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_flipout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_flipout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_flipout_1/strided_sliceStridedSliceconv2d_flipout_1/Shape:output:0-conv2d_flipout_1/strided_slice/stack:output:0/conv2d_flipout_1/strided_slice/stack_1:output:0/conv2d_flipout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
conv2d_flipout_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/ExpandDims
ExpandDims'conv2d_flipout_1/strided_slice:output:0(conv2d_flipout_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:y
&conv2d_flipout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????r
(conv2d_flipout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_flipout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_flipout_1/strided_slice_1StridedSliceconv2d_flipout_1/Shape:output:0/conv2d_flipout_1/strided_slice_1/stack:output:01conv2d_flipout_1/strided_slice_1/stack_1:output:01conv2d_flipout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!conv2d_flipout_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/ExpandDims_1
ExpandDims)conv2d_flipout_1/strided_slice_1:output:0*conv2d_flipout_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:^
conv2d_flipout_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/concatConcatV2$conv2d_flipout_1/ExpandDims:output:0&conv2d_flipout_1/ExpandDims_1:output:0%conv2d_flipout_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
<conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
:conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
6conv2d_flipout_1/rademacher/uniform/sanitize_seed/seedRandomUniformIntEconv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/shape:output:0Cconv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/min:output:0Cconv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
@conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
@conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Yconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter?conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
@conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
<conv2d_flipout_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2 conv2d_flipout_1/concat:output:0_conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0cconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Iconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/alg:output:0Iconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/min:output:0Iconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	c
!conv2d_flipout_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout_1/rademacher/mulMul*conv2d_flipout_1/rademacher/mul/x:output:0Econv2d_flipout_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? c
!conv2d_flipout_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout_1/rademacher/subSub#conv2d_flipout_1/rademacher/mul:z:0*conv2d_flipout_1/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? ?
 conv2d_flipout_1/rademacher/CastCast#conv2d_flipout_1/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? e
#conv2d_flipout_1/ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B :@c
!conv2d_flipout_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/ExpandDims_2
ExpandDims,conv2d_flipout_1/ExpandDims_2/input:output:0*conv2d_flipout_1/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:`
conv2d_flipout_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/concat_1ConcatV2$conv2d_flipout_1/ExpandDims:output:0&conv2d_flipout_1/ExpandDims_2:output:0'conv2d_flipout_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
>conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
8conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntGconv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Econv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0Econv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Bconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Bconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
[conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterAconv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Bconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
>conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2"conv2d_flipout_1/concat_1:output:0aconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0econv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Kconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Kconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Kconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????@*
dtype0	e
#conv2d_flipout_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
!conv2d_flipout_1/rademacher_1/mulMul,conv2d_flipout_1/rademacher_1/mul/x:output:0Gconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????@e
#conv2d_flipout_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
!conv2d_flipout_1/rademacher_1/subSub%conv2d_flipout_1/rademacher_1/mul:z:0,conv2d_flipout_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????@?
"conv2d_flipout_1/rademacher_1/CastCast%conv2d_flipout_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????@c
!conv2d_flipout_1/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_3
ExpandDims$conv2d_flipout_1/rademacher/Cast:y:0*conv2d_flipout_1/ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:????????? c
!conv2d_flipout_1/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_4
ExpandDims&conv2d_flipout_1/rademacher_1/Cast:y:0*conv2d_flipout_1/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:?????????@c
!conv2d_flipout_1/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_5
ExpandDims&conv2d_flipout_1/ExpandDims_3:output:0*conv2d_flipout_1/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:????????? c
!conv2d_flipout_1/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_6
ExpandDims&conv2d_flipout_1/ExpandDims_4:output:0*conv2d_flipout_1/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_flipout_1/mulMulactivation/Relu:activations:0&conv2d_flipout_1/ExpandDims_5:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_flipout_1/Conv2D_1Conv2Dconv2d_flipout_1/mul:z:0/conv2d_flipout_1/Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_flipout_1/mul_1Mul"conv2d_flipout_1/Conv2D_1:output:0&conv2d_flipout_1/ExpandDims_6:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_flipout_1/addAddV2 conv2d_flipout_1/Conv2D:output:0conv2d_flipout_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@?
=conv2d_flipout_1/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Rconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
cconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Tconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOp]conv2d_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:@*
dtype0?
Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:@?
Kconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Yconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
[conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
[conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Sconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0bconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0dconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0dconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Vconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Xconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Sconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgsaconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0\conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Qconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Xconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Zconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Qconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastTo\conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:@?
Sconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
Mconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeZconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0\conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:@?
>conv2d_flipout_1/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:@?
8conv2d_flipout_1/IndependentDeterministic/sample/ReshapeReshapeVconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Gconv2d_flipout_1/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:@?
conv2d_flipout_1/BiasAddBiasAddconv2d_flipout_1/add:z:0Aconv2d_flipout_1/IndependentDeterministic/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp?conv2d_flipout_1_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559354*
T0*
_output_shapes
: ?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp/conv2d_flipout_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559354*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559354*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
"conv2d_flipout_1/divergence_kernelIdentity?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: ?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!conv2d_flipout_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
$batch_normalization_1/AssignNewValueAssignVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
&batch_normalization_1/AssignNewValue_1AssignVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten/ReshapeReshapeactivation_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????y
(dense_flipout/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     c
dense_flipout/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_flipout/zeros_likeFill1dense_flipout/zeros_like/shape_as_tensor:output:0'dense_flipout/zeros_like/Const:output:0*
T0* 
_output_shapes
:
??k
(dense_flipout/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
3dense_flipout/Normal/sample/Softplus/ReadVariableOpReadVariableOp<dense_flipout_normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
$dense_flipout/Normal/sample/SoftplusSoftplus;dense_flipout/Normal/sample/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??f
!dense_flipout/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
dense_flipout/Normal/sample/addAddV2*dense_flipout/Normal/sample/add/x:output:02dense_flipout/Normal/sample/Softplus:activations:0*
T0* 
_output_shapes
:
??|
+dense_flipout/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     c
!dense_flipout/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : y
/dense_flipout/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_flipout/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_flipout/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_flipout/Normal/sample/strided_sliceStridedSlice4dense_flipout/Normal/sample/shape_as_tensor:output:08dense_flipout/Normal/sample/strided_slice/stack:output:0:dense_flipout/Normal/sample/strided_slice/stack_1:output:0:dense_flipout/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
-dense_flipout/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"@     e
#dense_flipout/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : {
1dense_flipout/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_flipout/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_flipout/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_flipout/Normal/sample/strided_slice_1StridedSlice6dense_flipout/Normal/sample/shape_as_tensor_1:output:0:dense_flipout/Normal/sample/strided_slice_1/stack:output:0<dense_flipout/Normal/sample/strided_slice_1/stack_1:output:0<dense_flipout/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masko
,dense_flipout/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB q
.dense_flipout/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
)dense_flipout/Normal/sample/BroadcastArgsBroadcastArgs7dense_flipout/Normal/sample/BroadcastArgs/s0_1:output:02dense_flipout/Normal/sample/strided_slice:output:0*
_output_shapes
:?
+dense_flipout/Normal/sample/BroadcastArgs_1BroadcastArgs.dense_flipout/Normal/sample/BroadcastArgs:r0:04dense_flipout/Normal/sample/strided_slice_1:output:0*
_output_shapes
:u
+dense_flipout/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
'dense_flipout/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"dense_flipout/Normal/sample/concatConcatV24dense_flipout/Normal/sample/concat/values_0:output:00dense_flipout/Normal/sample/BroadcastArgs_1:r0:00dense_flipout/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:z
5dense_flipout/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    |
7dense_flipout/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Edense_flipout/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal+dense_flipout/Normal/sample/concat:output:0*
T0*$
_output_shapes
:??*
dtype0?
4dense_flipout/Normal/sample/normal/random_normal/mulMulNdense_flipout/Normal/sample/normal/random_normal/RandomStandardNormal:output:0@dense_flipout/Normal/sample/normal/random_normal/stddev:output:0*
T0*$
_output_shapes
:???
0dense_flipout/Normal/sample/normal/random_normalAddV28dense_flipout/Normal/sample/normal/random_normal/mul:z:0>dense_flipout/Normal/sample/normal/random_normal/mean:output:0*
T0*$
_output_shapes
:???
dense_flipout/Normal/sample/mulMul4dense_flipout/Normal/sample/normal/random_normal:z:0#dense_flipout/Normal/sample/add:z:0*
T0*$
_output_shapes
:???
!dense_flipout/Normal/sample/add_1AddV2#dense_flipout/Normal/sample/mul:z:0!dense_flipout/zeros_like:output:0*
T0*$
_output_shapes
:??z
)dense_flipout/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     ?
#dense_flipout/Normal/sample/ReshapeReshape%dense_flipout/Normal/sample/add_1:z:02dense_flipout/Normal/sample/Reshape/shape:output:0*
T0* 
_output_shapes
:
??[
dense_flipout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:k
!dense_flipout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#dense_flipout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#dense_flipout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_flipout/strided_sliceStridedSlicedense_flipout/Shape:output:0*dense_flipout/strided_slice/stack:output:0,dense_flipout/strided_slice/stack_1:output:0,dense_flipout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9dense_flipout/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
7dense_flipout/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????}
7dense_flipout/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
3dense_flipout/rademacher/uniform/sanitize_seed/seedRandomUniformIntBdense_flipout/rademacher/uniform/sanitize_seed/seed/shape:output:0@dense_flipout/rademacher/uniform/sanitize_seed/seed/min:output:0@dense_flipout/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
=dense_flipout/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
=dense_flipout/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Vdense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter<dense_flipout/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
=dense_flipout/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
9dense_flipout/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout/Shape:output:0\dense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`dense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Fdense_flipout/rademacher/uniform/stateless_random_uniform/alg:output:0Fdense_flipout/rademacher/uniform/stateless_random_uniform/min:output:0Fdense_flipout/rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	`
dense_flipout/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher/mulMul'dense_flipout/rademacher/mul/x:output:0Bdense_flipout/rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????`
dense_flipout/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher/subSub dense_flipout/rademacher/mul:z:0'dense_flipout/rademacher/sub/y:output:0*
T0	*(
_output_shapes
:???????????
dense_flipout/rademacher/CastCast dense_flipout/rademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????a
dense_flipout/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value
B :?^
dense_flipout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout/ExpandDims
ExpandDims'dense_flipout/ExpandDims/input:output:0%dense_flipout/ExpandDims/dim:output:0*
T0*
_output_shapes
:[
dense_flipout/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout/concatConcatV2$dense_flipout/strided_slice:output:0!dense_flipout/ExpandDims:output:0"dense_flipout/concat/axis:output:0*
N*
T0*
_output_shapes
:?
;dense_flipout/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9dense_flipout/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????
9dense_flipout/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
5dense_flipout/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntDdense_flipout/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Bdense_flipout/rademacher_1/uniform/sanitize_seed/seed/min:output:0Bdense_flipout/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
?dense_flipout/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
?dense_flipout/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Xdense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter>dense_flipout/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
?dense_flipout/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
;dense_flipout/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout/concat:output:0^dense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0bdense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hdense_flipout/rademacher_1/uniform/stateless_random_uniform/alg:output:0Hdense_flipout/rademacher_1/uniform/stateless_random_uniform/min:output:0Hdense_flipout/rademacher_1/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	b
 dense_flipout/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher_1/mulMul)dense_flipout/rademacher_1/mul/x:output:0Ddense_flipout/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????b
 dense_flipout/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher_1/subSub"dense_flipout/rademacher_1/mul:z:0)dense_flipout/rademacher_1/sub/y:output:0*
T0	*(
_output_shapes
:???????????
dense_flipout/rademacher_1/CastCast"dense_flipout/rademacher_1/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:???????????
dense_flipout/mulMulflatten/Reshape:output:0!dense_flipout/rademacher/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_flipout/MatMulMatMuldense_flipout/mul:z:0,dense_flipout/Normal/sample/Reshape:output:0*
T0*(
_output_shapes
:???????????
dense_flipout/mul_1Muldense_flipout/MatMul:product:0#dense_flipout/rademacher_1/Cast:y:0*
T0*(
_output_shapes
:???????????
%dense_flipout/MatMul_1/ReadVariableOpReadVariableOp.dense_flipout_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_flipout/MatMul_1MatMulflatten/Reshape:output:0-dense_flipout/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_flipout/addAddV2 dense_flipout/MatMul_1:product:0dense_flipout/mul_1:z:0*
T0*(
_output_shapes
:??????????}
:dense_flipout/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Odense_flipout/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
`dense_flipout/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Qdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpZdense_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:??
Hdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Vdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Xdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Xdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Pdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice[dense_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0_dense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0adense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0adense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Sdense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Udense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Pdense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs^dense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Ydense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Ndense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Idense_flipout/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2[dense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Udense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0[dense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Wdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ndense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToYdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes
:	??
Pdense_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Jdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeWdense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Ydense_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	??
;dense_flipout/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
5dense_flipout/IndependentDeterministic/sample/ReshapeReshapeSdense_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Ddense_flipout/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes	
:??
dense_flipout/BiasAddBiasAdddense_flipout/add:z:0>dense_flipout/IndependentDeterministic/sample/Reshape:output:0*
T0*(
_output_shapes
:??????????m
dense_flipout/ReluReludense_flipout/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp<dense_flipout_normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559506*
T0*
_output_shapes
: ?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp.dense_flipout_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559506*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559506*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
dense_flipout/divergence_kernelIdentity?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: {
*dense_flipout_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   e
 dense_flipout_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_flipout_1/zeros_likeFill3dense_flipout_1/zeros_like/shape_as_tensor:output:0)dense_flipout_1/zeros_like/Const:output:0*
T0*
_output_shapes
:	?
m
*dense_flipout_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
5dense_flipout_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp>dense_flipout_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
&dense_flipout_1/Normal/sample/SoftplusSoftplus=dense_flipout_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
h
#dense_flipout_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
!dense_flipout_1/Normal/sample/addAddV2,dense_flipout_1/Normal/sample/add/x:output:04dense_flipout_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	?
~
-dense_flipout_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   e
#dense_flipout_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : {
1dense_flipout_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_flipout_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_flipout_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_flipout_1/Normal/sample/strided_sliceStridedSlice6dense_flipout_1/Normal/sample/shape_as_tensor:output:0:dense_flipout_1/Normal/sample/strided_slice/stack:output:0<dense_flipout_1/Normal/sample/strided_slice/stack_1:output:0<dense_flipout_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
/dense_flipout_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   
   g
%dense_flipout_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : }
3dense_flipout_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_flipout_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_flipout_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_flipout_1/Normal/sample/strided_slice_1StridedSlice8dense_flipout_1/Normal/sample/shape_as_tensor_1:output:0<dense_flipout_1/Normal/sample/strided_slice_1/stack:output:0>dense_flipout_1/Normal/sample/strided_slice_1/stack_1:output:0>dense_flipout_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskq
.dense_flipout_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB s
0dense_flipout_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
+dense_flipout_1/Normal/sample/BroadcastArgsBroadcastArgs9dense_flipout_1/Normal/sample/BroadcastArgs/s0_1:output:04dense_flipout_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
-dense_flipout_1/Normal/sample/BroadcastArgs_1BroadcastArgs0dense_flipout_1/Normal/sample/BroadcastArgs:r0:06dense_flipout_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:w
-dense_flipout_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:k
)dense_flipout_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$dense_flipout_1/Normal/sample/concatConcatV26dense_flipout_1/Normal/sample/concat/values_0:output:02dense_flipout_1/Normal/sample/BroadcastArgs_1:r0:02dense_flipout_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:|
7dense_flipout_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
9dense_flipout_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Gdense_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal-dense_flipout_1/Normal/sample/concat:output:0*
T0*#
_output_shapes
:?
*
dtype0?
6dense_flipout_1/Normal/sample/normal/random_normal/mulMulPdense_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Bdense_flipout_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:?
?
2dense_flipout_1/Normal/sample/normal/random_normalAddV2:dense_flipout_1/Normal/sample/normal/random_normal/mul:z:0@dense_flipout_1/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:?
?
!dense_flipout_1/Normal/sample/mulMul6dense_flipout_1/Normal/sample/normal/random_normal:z:0%dense_flipout_1/Normal/sample/add:z:0*
T0*#
_output_shapes
:?
?
#dense_flipout_1/Normal/sample/add_1AddV2%dense_flipout_1/Normal/sample/mul:z:0#dense_flipout_1/zeros_like:output:0*
T0*#
_output_shapes
:?
|
+dense_flipout_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
%dense_flipout_1/Normal/sample/ReshapeReshape'dense_flipout_1/Normal/sample/add_1:z:04dense_flipout_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	?
e
dense_flipout_1/ShapeShape dense_flipout/Relu:activations:0*
T0*
_output_shapes
:m
#dense_flipout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%dense_flipout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%dense_flipout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_flipout_1/strided_sliceStridedSlicedense_flipout_1/Shape:output:0,dense_flipout_1/strided_slice/stack:output:0.dense_flipout_1/strided_slice/stack_1:output:0.dense_flipout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
;dense_flipout_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9dense_flipout_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????
9dense_flipout_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
5dense_flipout_1/rademacher/uniform/sanitize_seed/seedRandomUniformIntDdense_flipout_1/rademacher/uniform/sanitize_seed/seed/shape:output:0Bdense_flipout_1/rademacher/uniform/sanitize_seed/seed/min:output:0Bdense_flipout_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
?dense_flipout_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
?dense_flipout_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Xdense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter>dense_flipout_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
?dense_flipout_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
;dense_flipout_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout_1/Shape:output:0^dense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0bdense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hdense_flipout_1/rademacher/uniform/stateless_random_uniform/alg:output:0Hdense_flipout_1/rademacher/uniform/stateless_random_uniform/min:output:0Hdense_flipout_1/rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	b
 dense_flipout_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout_1/rademacher/mulMul)dense_flipout_1/rademacher/mul/x:output:0Ddense_flipout_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????b
 dense_flipout_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout_1/rademacher/subSub"dense_flipout_1/rademacher/mul:z:0)dense_flipout_1/rademacher/sub/y:output:0*
T0	*(
_output_shapes
:???????????
dense_flipout_1/rademacher/CastCast"dense_flipout_1/rademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????b
 dense_flipout_1/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :
`
dense_flipout_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout_1/ExpandDims
ExpandDims)dense_flipout_1/ExpandDims/input:output:0'dense_flipout_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:]
dense_flipout_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout_1/concatConcatV2&dense_flipout_1/strided_slice:output:0#dense_flipout_1/ExpandDims:output:0$dense_flipout_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
=dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
;dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
;dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
7dense_flipout_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntFdense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Ddense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0Ddense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Adense_flipout_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Adense_flipout_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Zdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter@dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Adense_flipout_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
=dense_flipout_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout_1/concat:output:0`dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ddense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Jdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Jdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Jdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????
*
dtype0	d
"dense_flipout_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
 dense_flipout_1/rademacher_1/mulMul+dense_flipout_1/rademacher_1/mul/x:output:0Fdense_flipout_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????
d
"dense_flipout_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
 dense_flipout_1/rademacher_1/subSub$dense_flipout_1/rademacher_1/mul:z:0+dense_flipout_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
?
!dense_flipout_1/rademacher_1/CastCast$dense_flipout_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????
?
dense_flipout_1/mulMul dense_flipout/Relu:activations:0#dense_flipout_1/rademacher/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_flipout_1/MatMulMatMuldense_flipout_1/mul:z:0.dense_flipout_1/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
dense_flipout_1/mul_1Mul dense_flipout_1/MatMul:product:0%dense_flipout_1/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????
?
'dense_flipout_1/MatMul_1/ReadVariableOpReadVariableOp0dense_flipout_1_matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_flipout_1/MatMul_1MatMul dense_flipout/Relu:activations:0/dense_flipout_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_flipout_1/addAddV2"dense_flipout_1/MatMul_1:product:0dense_flipout_1/mul_1:z:0*
T0*'
_output_shapes
:?????????

<dense_flipout_1/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Qdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
bdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Sdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOp\dense_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:
*
dtype0?
Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
?
Jdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Xdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Zdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Zdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Rdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice]dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0adense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0cdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0cdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Udense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Wdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Rdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs`dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0[dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Pdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Kdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2]dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Wdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0]dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Ydense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Pdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastTo[dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:
?
Rdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
Ldense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeYdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0[dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
?
=dense_flipout_1/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
7dense_flipout_1/IndependentDeterministic/sample/ReshapeReshapeUdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Fdense_flipout_1/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:
?
dense_flipout_1/BiasAddBiasAdddense_flipout_1/add:z:0@dense_flipout_1/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp>dense_flipout_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559640*
T0*
_output_shapes
: ?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp0dense_flipout_1_matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559640*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559640*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
!dense_flipout_1/divergence_kernelIdentity?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity dense_flipout_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)conv2d_flipout/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2d_flipout_1/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(dense_flipout/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: j

Identity_4Identity*dense_flipout_1/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^conv2d_flipout/Conv2D/ReadVariableOpS^conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp5^conv2d_flipout/Normal/sample/Softplus/ReadVariableOp'^conv2d_flipout_1/Conv2D/ReadVariableOpU^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp7^conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOpR^dense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp&^dense_flipout/MatMul_1/ReadVariableOp4^dense_flipout/Normal/sample/Softplus/ReadVariableOpT^dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp(^dense_flipout_1/MatMul_1/ReadVariableOp6^dense_flipout_1/Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$conv2d_flipout/Conv2D/ReadVariableOp$conv2d_flipout/Conv2D/ReadVariableOp2?
Rconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpRconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2l
4conv2d_flipout/Normal/sample/Softplus/ReadVariableOp4conv2d_flipout/Normal/sample/Softplus/ReadVariableOp2P
&conv2d_flipout_1/Conv2D/ReadVariableOp&conv2d_flipout_1/Conv2D/ReadVariableOp2?
Tconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpTconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2p
6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp2?
Qdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpQdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2N
%dense_flipout/MatMul_1/ReadVariableOp%dense_flipout/MatMul_1/ReadVariableOp2j
3dense_flipout/Normal/sample/Softplus/ReadVariableOp3dense_flipout/Normal/sample/Softplus/ReadVariableOp2?
Sdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpSdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2R
'dense_flipout_1/MatMul_1/ReadVariableOp'dense_flipout_1/MatMul_1/ReadVariableOp2n
5dense_flipout_1/Normal/sample/Softplus/ReadVariableOp5dense_flipout_1/Normal/sample/Softplus/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
E
)__inference_flatten_layer_call_fn_2560227

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2557432a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??

?:
"__inference__wrapped_model_2556921
conv2d_flipout_inputb
Hsequential_conv2d_flipout_normal_sample_softplus_readvariableop_resource: R
8sequential_conv2d_flipout_conv2d_readvariableop_resource: ?
?sequential_conv2d_flipout_independentdeterministic_constructed_at_conv2d_flipout_sample_deterministic_sample_readvariableop_resource: ?
?sequential_conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556441?
?sequential_conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xD
6sequential_batch_normalization_readvariableop_resource: F
8sequential_batch_normalization_readvariableop_1_resource: U
Gsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource: W
Isequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource: d
Jsequential_conv2d_flipout_1_normal_sample_softplus_readvariableop_resource: @T
:sequential_conv2d_flipout_1_conv2d_readvariableop_resource: @?
?sequential_conv2d_flipout_1_independentdeterministic_constructed_at_conv2d_flipout_1_sample_deterministic_sample_readvariableop_resource:@?
?sequential_conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556608?
?sequential_conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xF
8sequential_batch_normalization_1_readvariableop_resource:@H
:sequential_batch_normalization_1_readvariableop_1_resource:@W
Isequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@Y
Ksequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@[
Gsequential_dense_flipout_normal_sample_softplus_readvariableop_resource:
??M
9sequential_dense_flipout_matmul_1_readvariableop_resource:
???
?sequential_dense_flipout_independentdeterministic_constructed_at_dense_flipout_sample_deterministic_sample_readvariableop_resource:	??
?sequential_dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556760?
?sequential_dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x\
Isequential_dense_flipout_1_normal_sample_softplus_readvariableop_resource:	?
N
;sequential_dense_flipout_1_matmul_1_readvariableop_resource:	?
?
?sequential_dense_flipout_1_independentdeterministic_constructed_at_dense_flipout_1_sample_deterministic_sample_readvariableop_resource:
?
?sequential_dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556894?
?sequential_dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity??>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp?@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1?-sequential/batch_normalization/ReadVariableOp?/sequential/batch_normalization/ReadVariableOp_1?@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp?Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?/sequential/batch_normalization_1/ReadVariableOp?1sequential/batch_normalization_1/ReadVariableOp_1?/sequential/conv2d_flipout/Conv2D/ReadVariableOp?{sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp??sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp??sequential/conv2d_flipout/Normal/sample/Softplus/ReadVariableOp?1sequential/conv2d_flipout_1/Conv2D/ReadVariableOp?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp??sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?Asequential/conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp?ysequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp??sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?0sequential/dense_flipout/MatMul_1/ReadVariableOp?>sequential/dense_flipout/Normal/sample/Softplus/ReadVariableOp?}sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp??sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?2sequential/dense_flipout_1/MatMul_1/ReadVariableOp?@sequential/dense_flipout_1/Normal/sample/Softplus/ReadVariableOp?
$sequential/conv2d_flipout/zeros_likeConst*&
_output_shapes
: *
dtype0*%
valueB *    w
4sequential/conv2d_flipout/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
?sequential/conv2d_flipout/Normal/sample/Softplus/ReadVariableOpReadVariableOpHsequential_conv2d_flipout_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
0sequential/conv2d_flipout/Normal/sample/SoftplusSoftplusGsequential/conv2d_flipout/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: r
-sequential/conv2d_flipout/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
+sequential/conv2d_flipout/Normal/sample/addAddV26sequential/conv2d_flipout/Normal/sample/add/x:output:0>sequential/conv2d_flipout/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: ?
7sequential/conv2d_flipout/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             o
-sequential/conv2d_flipout/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
;sequential/conv2d_flipout/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
=sequential/conv2d_flipout/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
=sequential/conv2d_flipout/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
5sequential/conv2d_flipout/Normal/sample/strided_sliceStridedSlice@sequential/conv2d_flipout/Normal/sample/shape_as_tensor:output:0Dsequential/conv2d_flipout/Normal/sample/strided_slice/stack:output:0Fsequential/conv2d_flipout/Normal/sample/strided_slice/stack_1:output:0Fsequential/conv2d_flipout/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9sequential/conv2d_flipout/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             q
/sequential/conv2d_flipout/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
=sequential/conv2d_flipout/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential/conv2d_flipout/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/conv2d_flipout/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential/conv2d_flipout/Normal/sample/strided_slice_1StridedSliceBsequential/conv2d_flipout/Normal/sample/shape_as_tensor_1:output:0Fsequential/conv2d_flipout/Normal/sample/strided_slice_1/stack:output:0Hsequential/conv2d_flipout/Normal/sample/strided_slice_1/stack_1:output:0Hsequential/conv2d_flipout/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask{
8sequential/conv2d_flipout/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB }
:sequential/conv2d_flipout/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
5sequential/conv2d_flipout/Normal/sample/BroadcastArgsBroadcastArgsCsequential/conv2d_flipout/Normal/sample/BroadcastArgs/s0_1:output:0>sequential/conv2d_flipout/Normal/sample/strided_slice:output:0*
_output_shapes
:?
7sequential/conv2d_flipout/Normal/sample/BroadcastArgs_1BroadcastArgs:sequential/conv2d_flipout/Normal/sample/BroadcastArgs:r0:0@sequential/conv2d_flipout/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
7sequential/conv2d_flipout/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:u
3sequential/conv2d_flipout/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
.sequential/conv2d_flipout/Normal/sample/concatConcatV2@sequential/conv2d_flipout/Normal/sample/concat/values_0:output:0<sequential/conv2d_flipout/Normal/sample/BroadcastArgs_1:r0:0<sequential/conv2d_flipout/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Asequential/conv2d_flipout/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Csequential/conv2d_flipout/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Qsequential/conv2d_flipout/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal7sequential/conv2d_flipout/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0?
@sequential/conv2d_flipout/Normal/sample/normal/random_normal/mulMulZsequential/conv2d_flipout/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Lsequential/conv2d_flipout/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: ?
<sequential/conv2d_flipout/Normal/sample/normal/random_normalAddV2Dsequential/conv2d_flipout/Normal/sample/normal/random_normal/mul:z:0Jsequential/conv2d_flipout/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: ?
+sequential/conv2d_flipout/Normal/sample/mulMul@sequential/conv2d_flipout/Normal/sample/normal/random_normal:z:0/sequential/conv2d_flipout/Normal/sample/add:z:0*
T0**
_output_shapes
: ?
-sequential/conv2d_flipout/Normal/sample/add_1AddV2/sequential/conv2d_flipout/Normal/sample/mul:z:0-sequential/conv2d_flipout/zeros_like:output:0*
T0**
_output_shapes
: ?
5sequential/conv2d_flipout/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
/sequential/conv2d_flipout/Normal/sample/ReshapeReshape1sequential/conv2d_flipout/Normal/sample/add_1:z:0>sequential/conv2d_flipout/Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: ?
/sequential/conv2d_flipout/Conv2D/ReadVariableOpReadVariableOp8sequential_conv2d_flipout_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
 sequential/conv2d_flipout/Conv2DConv2Dconv2d_flipout_input7sequential/conv2d_flipout/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
c
sequential/conv2d_flipout/ShapeShapeconv2d_flipout_input*
T0*
_output_shapes
:w
-sequential/conv2d_flipout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/sequential/conv2d_flipout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/sequential/conv2d_flipout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
'sequential/conv2d_flipout/strided_sliceStridedSlice(sequential/conv2d_flipout/Shape:output:06sequential/conv2d_flipout/strided_slice/stack:output:08sequential/conv2d_flipout/strided_slice/stack_1:output:08sequential/conv2d_flipout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
(sequential/conv2d_flipout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
$sequential/conv2d_flipout/ExpandDims
ExpandDims0sequential/conv2d_flipout/strided_slice:output:01sequential/conv2d_flipout/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
/sequential/conv2d_flipout/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????{
1sequential/conv2d_flipout/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: {
1sequential/conv2d_flipout/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/conv2d_flipout/strided_slice_1StridedSlice(sequential/conv2d_flipout/Shape:output:08sequential/conv2d_flipout/strided_slice_1/stack:output:0:sequential/conv2d_flipout/strided_slice_1/stack_1:output:0:sequential/conv2d_flipout/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential/conv2d_flipout/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential/conv2d_flipout/ExpandDims_1
ExpandDims2sequential/conv2d_flipout/strided_slice_1:output:03sequential/conv2d_flipout/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:g
%sequential/conv2d_flipout/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
 sequential/conv2d_flipout/concatConcatV2-sequential/conv2d_flipout/ExpandDims:output:0/sequential/conv2d_flipout/ExpandDims_1:output:0.sequential/conv2d_flipout/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Esequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Csequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Csequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
?sequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seedRandomUniformIntNsequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seed/shape:output:0Lsequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seed/min:output:0Lsequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Isequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Isequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
bsequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterHsequential/conv2d_flipout/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Isequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Esequential/conv2d_flipout/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2)sequential/conv2d_flipout/concat:output:0hsequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0lsequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Rsequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/alg:output:0Rsequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/min:output:0Rsequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	l
*sequential/conv2d_flipout/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
(sequential/conv2d_flipout/rademacher/mulMul3sequential/conv2d_flipout/rademacher/mul/x:output:0Nsequential/conv2d_flipout/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????l
*sequential/conv2d_flipout/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
(sequential/conv2d_flipout/rademacher/subSub,sequential/conv2d_flipout/rademacher/mul:z:03sequential/conv2d_flipout/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:??????????
)sequential/conv2d_flipout/rademacher/CastCast,sequential/conv2d_flipout/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????n
,sequential/conv2d_flipout/ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B : l
*sequential/conv2d_flipout/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential/conv2d_flipout/ExpandDims_2
ExpandDims5sequential/conv2d_flipout/ExpandDims_2/input:output:03sequential/conv2d_flipout/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:i
'sequential/conv2d_flipout/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"sequential/conv2d_flipout/concat_1ConcatV2-sequential/conv2d_flipout/ExpandDims:output:0/sequential/conv2d_flipout/ExpandDims_2:output:00sequential/conv2d_flipout/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Gsequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Esequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Esequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
Asequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntPsequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Nsequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/min:output:0Nsequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Ksequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Ksequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dsequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterJsequential/conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Ksequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Gsequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2+sequential/conv2d_flipout/concat_1:output:0jsequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0nsequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Tsequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/alg:output:0Tsequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/min:output:0Tsequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	n
,sequential/conv2d_flipout/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
*sequential/conv2d_flipout/rademacher_1/mulMul5sequential/conv2d_flipout/rademacher_1/mul/x:output:0Psequential/conv2d_flipout/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? n
,sequential/conv2d_flipout/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
*sequential/conv2d_flipout/rademacher_1/subSub.sequential/conv2d_flipout/rademacher_1/mul:z:05sequential/conv2d_flipout/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? ?
+sequential/conv2d_flipout/rademacher_1/CastCast.sequential/conv2d_flipout/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? l
*sequential/conv2d_flipout/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential/conv2d_flipout/ExpandDims_3
ExpandDims-sequential/conv2d_flipout/rademacher/Cast:y:03sequential/conv2d_flipout/ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:?????????l
*sequential/conv2d_flipout/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential/conv2d_flipout/ExpandDims_4
ExpandDims/sequential/conv2d_flipout/rademacher_1/Cast:y:03sequential/conv2d_flipout/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:????????? l
*sequential/conv2d_flipout/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential/conv2d_flipout/ExpandDims_5
ExpandDims/sequential/conv2d_flipout/ExpandDims_3:output:03sequential/conv2d_flipout/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????l
*sequential/conv2d_flipout/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
&sequential/conv2d_flipout/ExpandDims_6
ExpandDims/sequential/conv2d_flipout/ExpandDims_4:output:03sequential/conv2d_flipout/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:????????? ?
sequential/conv2d_flipout/mulMulconv2d_flipout_input/sequential/conv2d_flipout/ExpandDims_5:output:0*
T0*/
_output_shapes
:??????????
"sequential/conv2d_flipout/Conv2D_1Conv2D!sequential/conv2d_flipout/mul:z:08sequential/conv2d_flipout/Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
sequential/conv2d_flipout/mul_1Mul+sequential/conv2d_flipout/Conv2D_1:output:0/sequential/conv2d_flipout/ExpandDims_6:output:0*
T0*/
_output_shapes
:????????? ?
sequential/conv2d_flipout/addAddV2)sequential/conv2d_flipout/Conv2D:output:0#sequential/conv2d_flipout/mul_1:z:0*
T0*/
_output_shapes
:????????? ?
dsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
ysequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
{sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOpReadVariableOp?sequential_conv2d_flipout_independentdeterministic_constructed_at_conv2d_flipout_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0?
|sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: ?
rsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
zsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_sliceStridedSlice?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/shape_as_tensor:output:0?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack:output:0?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_1:output:0?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
}sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
zsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgsBroadcastArgs?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
|sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
|sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
xsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
ssequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concatConcatV2?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_0:output:0sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs:r0:0?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_2:output:0?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
xsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastToBroadcastTo?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp:value:0|sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: ?
zsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ?
tsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReshapeReshape?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastTo:output:0?sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ?
esequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
_sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/ReshapeReshape}sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape:output:0nsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape/shape:output:0*
T0*
_output_shapes
: ?
!sequential/conv2d_flipout/BiasAddBiasAdd!sequential/conv2d_flipout/add:z:0hsequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpHsequential_conv2d_flipout_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?sequential_conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556441*
T0*
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp8sequential_conv2d_flipout_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?sequential_conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556441*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?sequential_conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?sequential_conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556441*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
+sequential/conv2d_flipout/divergence_kernelIdentity?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: ?
-sequential/batch_normalization/ReadVariableOpReadVariableOp6sequential_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
/sequential/batch_normalization/ReadVariableOp_1ReadVariableOp8sequential_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpGsequential_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpIsequential_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
/sequential/batch_normalization/FusedBatchNormV3FusedBatchNormV3*sequential/conv2d_flipout/BiasAdd:output:05sequential/batch_normalization/ReadVariableOp:value:07sequential/batch_normalization/ReadVariableOp_1:value:0Fsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Hsequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( ?
sequential/activation/ReluRelu3sequential/batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? ?
6sequential/conv2d_flipout_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   q
,sequential/conv2d_flipout_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
&sequential/conv2d_flipout_1/zeros_likeFill?sequential/conv2d_flipout_1/zeros_like/shape_as_tensor:output:05sequential/conv2d_flipout_1/zeros_like/Const:output:0*
T0*&
_output_shapes
: @y
6sequential/conv2d_flipout_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Asequential/conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpJsequential_conv2d_flipout_1_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
2sequential/conv2d_flipout_1/Normal/sample/SoftplusSoftplusIsequential/conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @t
/sequential/conv2d_flipout_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
-sequential/conv2d_flipout_1/Normal/sample/addAddV28sequential/conv2d_flipout_1/Normal/sample/add/x:output:0@sequential/conv2d_flipout_1/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: @?
9sequential/conv2d_flipout_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   q
/sequential/conv2d_flipout_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
=sequential/conv2d_flipout_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential/conv2d_flipout_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/conv2d_flipout_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
7sequential/conv2d_flipout_1/Normal/sample/strided_sliceStridedSliceBsequential/conv2d_flipout_1/Normal/sample/shape_as_tensor:output:0Fsequential/conv2d_flipout_1/Normal/sample/strided_slice/stack:output:0Hsequential/conv2d_flipout_1/Normal/sample/strided_slice/stack_1:output:0Hsequential/conv2d_flipout_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
;sequential/conv2d_flipout_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"          @   s
1sequential/conv2d_flipout_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/conv2d_flipout_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Asequential/conv2d_flipout_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Asequential/conv2d_flipout_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
9sequential/conv2d_flipout_1/Normal/sample/strided_slice_1StridedSliceDsequential/conv2d_flipout_1/Normal/sample/shape_as_tensor_1:output:0Hsequential/conv2d_flipout_1/Normal/sample/strided_slice_1/stack:output:0Jsequential/conv2d_flipout_1/Normal/sample/strided_slice_1/stack_1:output:0Jsequential/conv2d_flipout_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask}
:sequential/conv2d_flipout_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB 
<sequential/conv2d_flipout_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
7sequential/conv2d_flipout_1/Normal/sample/BroadcastArgsBroadcastArgsEsequential/conv2d_flipout_1/Normal/sample/BroadcastArgs/s0_1:output:0@sequential/conv2d_flipout_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
9sequential/conv2d_flipout_1/Normal/sample/BroadcastArgs_1BroadcastArgs<sequential/conv2d_flipout_1/Normal/sample/BroadcastArgs:r0:0Bsequential/conv2d_flipout_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
9sequential/conv2d_flipout_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:w
5sequential/conv2d_flipout_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
0sequential/conv2d_flipout_1/Normal/sample/concatConcatV2Bsequential/conv2d_flipout_1/Normal/sample/concat/values_0:output:0>sequential/conv2d_flipout_1/Normal/sample/BroadcastArgs_1:r0:0>sequential/conv2d_flipout_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Csequential/conv2d_flipout_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Esequential/conv2d_flipout_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Ssequential/conv2d_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal9sequential/conv2d_flipout_1/Normal/sample/concat:output:0*
T0**
_output_shapes
: @*
dtype0?
Bsequential/conv2d_flipout_1/Normal/sample/normal/random_normal/mulMul\sequential/conv2d_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Nsequential/conv2d_flipout_1/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: @?
>sequential/conv2d_flipout_1/Normal/sample/normal/random_normalAddV2Fsequential/conv2d_flipout_1/Normal/sample/normal/random_normal/mul:z:0Lsequential/conv2d_flipout_1/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: @?
-sequential/conv2d_flipout_1/Normal/sample/mulMulBsequential/conv2d_flipout_1/Normal/sample/normal/random_normal:z:01sequential/conv2d_flipout_1/Normal/sample/add:z:0*
T0**
_output_shapes
: @?
/sequential/conv2d_flipout_1/Normal/sample/add_1AddV21sequential/conv2d_flipout_1/Normal/sample/mul:z:0/sequential/conv2d_flipout_1/zeros_like:output:0*
T0**
_output_shapes
: @?
7sequential/conv2d_flipout_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   ?
1sequential/conv2d_flipout_1/Normal/sample/ReshapeReshape3sequential/conv2d_flipout_1/Normal/sample/add_1:z:0@sequential/conv2d_flipout_1/Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: @?
1sequential/conv2d_flipout_1/Conv2D/ReadVariableOpReadVariableOp:sequential_conv2d_flipout_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
"sequential/conv2d_flipout_1/Conv2DConv2D(sequential/activation/Relu:activations:09sequential/conv2d_flipout_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
y
!sequential/conv2d_flipout_1/ShapeShape(sequential/activation/Relu:activations:0*
T0*
_output_shapes
:y
/sequential/conv2d_flipout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1sequential/conv2d_flipout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1sequential/conv2d_flipout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)sequential/conv2d_flipout_1/strided_sliceStridedSlice*sequential/conv2d_flipout_1/Shape:output:08sequential/conv2d_flipout_1/strided_slice/stack:output:0:sequential/conv2d_flipout_1/strided_slice/stack_1:output:0:sequential/conv2d_flipout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
*sequential/conv2d_flipout_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
&sequential/conv2d_flipout_1/ExpandDims
ExpandDims2sequential/conv2d_flipout_1/strided_slice:output:03sequential/conv2d_flipout_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:?
1sequential/conv2d_flipout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????}
3sequential/conv2d_flipout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: }
3sequential/conv2d_flipout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+sequential/conv2d_flipout_1/strided_slice_1StridedSlice*sequential/conv2d_flipout_1/Shape:output:0:sequential/conv2d_flipout_1/strided_slice_1/stack:output:0<sequential/conv2d_flipout_1/strided_slice_1/stack_1:output:0<sequential/conv2d_flipout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
,sequential/conv2d_flipout_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential/conv2d_flipout_1/ExpandDims_1
ExpandDims4sequential/conv2d_flipout_1/strided_slice_1:output:05sequential/conv2d_flipout_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:i
'sequential/conv2d_flipout_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"sequential/conv2d_flipout_1/concatConcatV2/sequential/conv2d_flipout_1/ExpandDims:output:01sequential/conv2d_flipout_1/ExpandDims_1:output:00sequential/conv2d_flipout_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Gsequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Esequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Esequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
Asequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seedRandomUniformIntPsequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/shape:output:0Nsequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/min:output:0Nsequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Ksequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Ksequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dsequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterJsequential/conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Ksequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Gsequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2+sequential/conv2d_flipout_1/concat:output:0jsequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0nsequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Tsequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/alg:output:0Tsequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/min:output:0Tsequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	n
,sequential/conv2d_flipout_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
*sequential/conv2d_flipout_1/rademacher/mulMul5sequential/conv2d_flipout_1/rademacher/mul/x:output:0Psequential/conv2d_flipout_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? n
,sequential/conv2d_flipout_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
*sequential/conv2d_flipout_1/rademacher/subSub.sequential/conv2d_flipout_1/rademacher/mul:z:05sequential/conv2d_flipout_1/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? ?
+sequential/conv2d_flipout_1/rademacher/CastCast.sequential/conv2d_flipout_1/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? p
.sequential/conv2d_flipout_1/ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B :@n
,sequential/conv2d_flipout_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
(sequential/conv2d_flipout_1/ExpandDims_2
ExpandDims7sequential/conv2d_flipout_1/ExpandDims_2/input:output:05sequential/conv2d_flipout_1/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:k
)sequential/conv2d_flipout_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$sequential/conv2d_flipout_1/concat_1ConcatV2/sequential/conv2d_flipout_1/ExpandDims:output:01sequential/conv2d_flipout_1/ExpandDims_2:output:02sequential/conv2d_flipout_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
Isequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Gsequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Gsequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
Csequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntRsequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Psequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0Psequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Msequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Msequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
fsequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterLsequential/conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Msequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Isequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2-sequential/conv2d_flipout_1/concat_1:output:0lsequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0psequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Vsequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Vsequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Vsequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????@*
dtype0	p
.sequential/conv2d_flipout_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
,sequential/conv2d_flipout_1/rademacher_1/mulMul7sequential/conv2d_flipout_1/rademacher_1/mul/x:output:0Rsequential/conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????@p
.sequential/conv2d_flipout_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
,sequential/conv2d_flipout_1/rademacher_1/subSub0sequential/conv2d_flipout_1/rademacher_1/mul:z:07sequential/conv2d_flipout_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????@?
-sequential/conv2d_flipout_1/rademacher_1/CastCast0sequential/conv2d_flipout_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????@n
,sequential/conv2d_flipout_1/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/conv2d_flipout_1/ExpandDims_3
ExpandDims/sequential/conv2d_flipout_1/rademacher/Cast:y:05sequential/conv2d_flipout_1/ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:????????? n
,sequential/conv2d_flipout_1/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/conv2d_flipout_1/ExpandDims_4
ExpandDims1sequential/conv2d_flipout_1/rademacher_1/Cast:y:05sequential/conv2d_flipout_1/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:?????????@n
,sequential/conv2d_flipout_1/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/conv2d_flipout_1/ExpandDims_5
ExpandDims1sequential/conv2d_flipout_1/ExpandDims_3:output:05sequential/conv2d_flipout_1/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:????????? n
,sequential/conv2d_flipout_1/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
(sequential/conv2d_flipout_1/ExpandDims_6
ExpandDims1sequential/conv2d_flipout_1/ExpandDims_4:output:05sequential/conv2d_flipout_1/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:?????????@?
sequential/conv2d_flipout_1/mulMul(sequential/activation/Relu:activations:01sequential/conv2d_flipout_1/ExpandDims_5:output:0*
T0*/
_output_shapes
:????????? ?
$sequential/conv2d_flipout_1/Conv2D_1Conv2D#sequential/conv2d_flipout_1/mul:z:0:sequential/conv2d_flipout_1/Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
!sequential/conv2d_flipout_1/mul_1Mul-sequential/conv2d_flipout_1/Conv2D_1:output:01sequential/conv2d_flipout_1/ExpandDims_6:output:0*
T0*/
_output_shapes
:?????????@?
sequential/conv2d_flipout_1/addAddV2+sequential/conv2d_flipout_1/Conv2D:output:0%sequential/conv2d_flipout_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@?
hsequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
}sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOpReadVariableOp?sequential_conv2d_flipout_1_independentdeterministic_constructed_at_conv2d_flipout_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:@*
dtype0?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:@?
vsequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
~sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_sliceStridedSlice?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/shape_as_tensor:output:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack:output:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_1:output:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
~sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgs?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
|sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
wsequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concatConcatV2?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_0:output:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs:r0:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_2:output:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
|sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastToBroadcastTo?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp:value:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:@?
~sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
xsequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReshapeReshape?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastTo:output:0?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:@?
isequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:@?
csequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/ReshapeReshape?sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape:output:0rsequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
:@?
#sequential/conv2d_flipout_1/BiasAddBiasAdd#sequential/conv2d_flipout_1/add:z:0lsequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpJsequential_conv2d_flipout_1_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?sequential_conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556608*
T0*
_output_shapes
: ?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp:sequential_conv2d_flipout_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?sequential_conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556608*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?sequential_conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?sequential_conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556608*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
-sequential/conv2d_flipout_1/divergence_kernelIdentity?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: ?
/sequential/batch_normalization_1/ReadVariableOpReadVariableOp8sequential_batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
1sequential/batch_normalization_1/ReadVariableOp_1ReadVariableOp:sequential_batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpIsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpKsequential_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
1sequential/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3,sequential/conv2d_flipout_1/BiasAdd:output:07sequential/batch_normalization_1/ReadVariableOp:value:09sequential/batch_normalization_1/ReadVariableOp_1:value:0Hsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Jsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
sequential/activation_1/ReluRelu5sequential/batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
sequential/flatten/ReshapeReshape*sequential/activation_1/Relu:activations:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:???????????
3sequential/dense_flipout/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     n
)sequential/dense_flipout/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
#sequential/dense_flipout/zeros_likeFill<sequential/dense_flipout/zeros_like/shape_as_tensor:output:02sequential/dense_flipout/zeros_like/Const:output:0*
T0* 
_output_shapes
:
??v
3sequential/dense_flipout/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
>sequential/dense_flipout/Normal/sample/Softplus/ReadVariableOpReadVariableOpGsequential_dense_flipout_normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
/sequential/dense_flipout/Normal/sample/SoftplusSoftplusFsequential/dense_flipout/Normal/sample/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??q
,sequential/dense_flipout/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
*sequential/dense_flipout/Normal/sample/addAddV25sequential/dense_flipout/Normal/sample/add/x:output:0=sequential/dense_flipout/Normal/sample/Softplus:activations:0*
T0* 
_output_shapes
:
???
6sequential/dense_flipout/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     n
,sequential/dense_flipout/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
:sequential/dense_flipout/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
<sequential/dense_flipout/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
<sequential/dense_flipout/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
4sequential/dense_flipout/Normal/sample/strided_sliceStridedSlice?sequential/dense_flipout/Normal/sample/shape_as_tensor:output:0Csequential/dense_flipout/Normal/sample/strided_slice/stack:output:0Esequential/dense_flipout/Normal/sample/strided_slice/stack_1:output:0Esequential/dense_flipout/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
8sequential/dense_flipout/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"@     p
.sequential/dense_flipout/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/dense_flipout/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>sequential/dense_flipout/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential/dense_flipout/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential/dense_flipout/Normal/sample/strided_slice_1StridedSliceAsequential/dense_flipout/Normal/sample/shape_as_tensor_1:output:0Esequential/dense_flipout/Normal/sample/strided_slice_1/stack:output:0Gsequential/dense_flipout/Normal/sample/strided_slice_1/stack_1:output:0Gsequential/dense_flipout/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskz
7sequential/dense_flipout/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB |
9sequential/dense_flipout/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
4sequential/dense_flipout/Normal/sample/BroadcastArgsBroadcastArgsBsequential/dense_flipout/Normal/sample/BroadcastArgs/s0_1:output:0=sequential/dense_flipout/Normal/sample/strided_slice:output:0*
_output_shapes
:?
6sequential/dense_flipout/Normal/sample/BroadcastArgs_1BroadcastArgs9sequential/dense_flipout/Normal/sample/BroadcastArgs:r0:0?sequential/dense_flipout/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
6sequential/dense_flipout/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:t
2sequential/dense_flipout/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
-sequential/dense_flipout/Normal/sample/concatConcatV2?sequential/dense_flipout/Normal/sample/concat/values_0:output:0;sequential/dense_flipout/Normal/sample/BroadcastArgs_1:r0:0;sequential/dense_flipout/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
@sequential/dense_flipout/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Bsequential/dense_flipout/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Psequential/dense_flipout/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal6sequential/dense_flipout/Normal/sample/concat:output:0*
T0*$
_output_shapes
:??*
dtype0?
?sequential/dense_flipout/Normal/sample/normal/random_normal/mulMulYsequential/dense_flipout/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Ksequential/dense_flipout/Normal/sample/normal/random_normal/stddev:output:0*
T0*$
_output_shapes
:???
;sequential/dense_flipout/Normal/sample/normal/random_normalAddV2Csequential/dense_flipout/Normal/sample/normal/random_normal/mul:z:0Isequential/dense_flipout/Normal/sample/normal/random_normal/mean:output:0*
T0*$
_output_shapes
:???
*sequential/dense_flipout/Normal/sample/mulMul?sequential/dense_flipout/Normal/sample/normal/random_normal:z:0.sequential/dense_flipout/Normal/sample/add:z:0*
T0*$
_output_shapes
:???
,sequential/dense_flipout/Normal/sample/add_1AddV2.sequential/dense_flipout/Normal/sample/mul:z:0,sequential/dense_flipout/zeros_like:output:0*
T0*$
_output_shapes
:???
4sequential/dense_flipout/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     ?
.sequential/dense_flipout/Normal/sample/ReshapeReshape0sequential/dense_flipout/Normal/sample/add_1:z:0=sequential/dense_flipout/Normal/sample/Reshape/shape:output:0*
T0* 
_output_shapes
:
??q
sequential/dense_flipout/ShapeShape#sequential/flatten/Reshape:output:0*
T0*
_output_shapes
:v
,sequential/dense_flipout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
.sequential/dense_flipout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????x
.sequential/dense_flipout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
&sequential/dense_flipout/strided_sliceStridedSlice'sequential/dense_flipout/Shape:output:05sequential/dense_flipout/strided_slice/stack:output:07sequential/dense_flipout/strided_slice/stack_1:output:07sequential/dense_flipout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Dsequential/dense_flipout/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Bsequential/dense_flipout/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Bsequential/dense_flipout/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
>sequential/dense_flipout/rademacher/uniform/sanitize_seed/seedRandomUniformIntMsequential/dense_flipout/rademacher/uniform/sanitize_seed/seed/shape:output:0Ksequential/dense_flipout/rademacher/uniform/sanitize_seed/seed/min:output:0Ksequential/dense_flipout/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Hsequential/dense_flipout/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Hsequential/dense_flipout/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
asequential/dense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterGsequential/dense_flipout/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Hsequential/dense_flipout/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Dsequential/dense_flipout/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2'sequential/dense_flipout/Shape:output:0gsequential/dense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ksequential/dense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Qsequential/dense_flipout/rademacher/uniform/stateless_random_uniform/alg:output:0Qsequential/dense_flipout/rademacher/uniform/stateless_random_uniform/min:output:0Qsequential/dense_flipout/rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	k
)sequential/dense_flipout/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
'sequential/dense_flipout/rademacher/mulMul2sequential/dense_flipout/rademacher/mul/x:output:0Msequential/dense_flipout/rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????k
)sequential/dense_flipout/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
'sequential/dense_flipout/rademacher/subSub+sequential/dense_flipout/rademacher/mul:z:02sequential/dense_flipout/rademacher/sub/y:output:0*
T0	*(
_output_shapes
:???????????
(sequential/dense_flipout/rademacher/CastCast+sequential/dense_flipout/rademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????l
)sequential/dense_flipout/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value
B :?i
'sequential/dense_flipout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
#sequential/dense_flipout/ExpandDims
ExpandDims2sequential/dense_flipout/ExpandDims/input:output:00sequential/dense_flipout/ExpandDims/dim:output:0*
T0*
_output_shapes
:f
$sequential/dense_flipout/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
sequential/dense_flipout/concatConcatV2/sequential/dense_flipout/strided_slice:output:0,sequential/dense_flipout/ExpandDims:output:0-sequential/dense_flipout/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Fsequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Dsequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
@sequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntOsequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Msequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seed/min:output:0Msequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Jsequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Jsequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
csequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterIsequential/dense_flipout/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Jsequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Fsequential/dense_flipout/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2(sequential/dense_flipout/concat:output:0isequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0msequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ssequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/alg:output:0Ssequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/min:output:0Ssequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	m
+sequential/dense_flipout/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
)sequential/dense_flipout/rademacher_1/mulMul4sequential/dense_flipout/rademacher_1/mul/x:output:0Osequential/dense_flipout/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????m
+sequential/dense_flipout/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
)sequential/dense_flipout/rademacher_1/subSub-sequential/dense_flipout/rademacher_1/mul:z:04sequential/dense_flipout/rademacher_1/sub/y:output:0*
T0	*(
_output_shapes
:???????????
*sequential/dense_flipout/rademacher_1/CastCast-sequential/dense_flipout/rademacher_1/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:???????????
sequential/dense_flipout/mulMul#sequential/flatten/Reshape:output:0,sequential/dense_flipout/rademacher/Cast:y:0*
T0*(
_output_shapes
:???????????
sequential/dense_flipout/MatMulMatMul sequential/dense_flipout/mul:z:07sequential/dense_flipout/Normal/sample/Reshape:output:0*
T0*(
_output_shapes
:???????????
sequential/dense_flipout/mul_1Mul)sequential/dense_flipout/MatMul:product:0.sequential/dense_flipout/rademacher_1/Cast:y:0*
T0*(
_output_shapes
:???????????
0sequential/dense_flipout/MatMul_1/ReadVariableOpReadVariableOp9sequential_dense_flipout_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
!sequential/dense_flipout/MatMul_1MatMul#sequential/flatten/Reshape:output:08sequential/dense_flipout/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
sequential/dense_flipout/addAddV2+sequential/dense_flipout/MatMul_1:product:0"sequential/dense_flipout/mul_1:z:0*
T0*(
_output_shapes
:???????????
bsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
wsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
ysequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOpReadVariableOp?sequential_dense_flipout_independentdeterministic_constructed_at_dense_flipout_sample_deterministic_sample_readvariableop_resource*
_output_shapes	
:?*
dtype0?
zsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:??
psequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
~sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
xsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_sliceStridedSlice?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/shape_as_tensor:output:0?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack:output:0?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_1:output:0?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
{sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
}sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
xsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgsBroadcastArgs?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
zsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
zsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
vsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
qsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concatConcatV2?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_0:output:0}sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs:r0:0?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_2:output:0sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
vsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastToBroadcastTo?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp:value:0zsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes
:	??
xsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
rsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReshapeReshapesequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastTo:output:0?sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	??
csequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
]sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/ReshapeReshape{sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape:output:0lsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape/shape:output:0*
T0*
_output_shapes	
:??
 sequential/dense_flipout/BiasAddBiasAdd sequential/dense_flipout/add:z:0fsequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape:output:0*
T0*(
_output_shapes
:???????????
sequential/dense_flipout/ReluRelu)sequential/dense_flipout/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpGsequential_dense_flipout_normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?sequential_dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556760*
T0*
_output_shapes
: ?
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp9sequential_dense_flipout_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?sequential_dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556760*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?sequential_dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?sequential_dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556760*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
*sequential/dense_flipout/divergence_kernelIdentity?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: ?
5sequential/dense_flipout_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   p
+sequential/dense_flipout_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
%sequential/dense_flipout_1/zeros_likeFill>sequential/dense_flipout_1/zeros_like/shape_as_tensor:output:04sequential/dense_flipout_1/zeros_like/Const:output:0*
T0*
_output_shapes
:	?
x
5sequential/dense_flipout_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
@sequential/dense_flipout_1/Normal/sample/Softplus/ReadVariableOpReadVariableOpIsequential_dense_flipout_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
1sequential/dense_flipout_1/Normal/sample/SoftplusSoftplusHsequential/dense_flipout_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
s
.sequential/dense_flipout_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
,sequential/dense_flipout_1/Normal/sample/addAddV27sequential/dense_flipout_1/Normal/sample/add/x:output:0?sequential/dense_flipout_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	?
?
8sequential/dense_flipout_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   p
.sequential/dense_flipout_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
<sequential/dense_flipout_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
>sequential/dense_flipout_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
>sequential/dense_flipout_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
6sequential/dense_flipout_1/Normal/sample/strided_sliceStridedSliceAsequential/dense_flipout_1/Normal/sample/shape_as_tensor:output:0Esequential/dense_flipout_1/Normal/sample/strided_slice/stack:output:0Gsequential/dense_flipout_1/Normal/sample/strided_slice/stack_1:output:0Gsequential/dense_flipout_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
:sequential/dense_flipout_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   
   r
0sequential/dense_flipout_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ?
>sequential/dense_flipout_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
@sequential/dense_flipout_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
@sequential/dense_flipout_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
8sequential/dense_flipout_1/Normal/sample/strided_slice_1StridedSliceCsequential/dense_flipout_1/Normal/sample/shape_as_tensor_1:output:0Gsequential/dense_flipout_1/Normal/sample/strided_slice_1/stack:output:0Isequential/dense_flipout_1/Normal/sample/strided_slice_1/stack_1:output:0Isequential/dense_flipout_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask|
9sequential/dense_flipout_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ~
;sequential/dense_flipout_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
6sequential/dense_flipout_1/Normal/sample/BroadcastArgsBroadcastArgsDsequential/dense_flipout_1/Normal/sample/BroadcastArgs/s0_1:output:0?sequential/dense_flipout_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
8sequential/dense_flipout_1/Normal/sample/BroadcastArgs_1BroadcastArgs;sequential/dense_flipout_1/Normal/sample/BroadcastArgs:r0:0Asequential/dense_flipout_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:?
8sequential/dense_flipout_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:v
4sequential/dense_flipout_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
/sequential/dense_flipout_1/Normal/sample/concatConcatV2Asequential/dense_flipout_1/Normal/sample/concat/values_0:output:0=sequential/dense_flipout_1/Normal/sample/BroadcastArgs_1:r0:0=sequential/dense_flipout_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Bsequential/dense_flipout_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
Dsequential/dense_flipout_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Rsequential/dense_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal8sequential/dense_flipout_1/Normal/sample/concat:output:0*
T0*#
_output_shapes
:?
*
dtype0?
Asequential/dense_flipout_1/Normal/sample/normal/random_normal/mulMul[sequential/dense_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Msequential/dense_flipout_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:?
?
=sequential/dense_flipout_1/Normal/sample/normal/random_normalAddV2Esequential/dense_flipout_1/Normal/sample/normal/random_normal/mul:z:0Ksequential/dense_flipout_1/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:?
?
,sequential/dense_flipout_1/Normal/sample/mulMulAsequential/dense_flipout_1/Normal/sample/normal/random_normal:z:00sequential/dense_flipout_1/Normal/sample/add:z:0*
T0*#
_output_shapes
:?
?
.sequential/dense_flipout_1/Normal/sample/add_1AddV20sequential/dense_flipout_1/Normal/sample/mul:z:0.sequential/dense_flipout_1/zeros_like:output:0*
T0*#
_output_shapes
:?
?
6sequential/dense_flipout_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
0sequential/dense_flipout_1/Normal/sample/ReshapeReshape2sequential/dense_flipout_1/Normal/sample/add_1:z:0?sequential/dense_flipout_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	?
{
 sequential/dense_flipout_1/ShapeShape+sequential/dense_flipout/Relu:activations:0*
T0*
_output_shapes
:x
.sequential/dense_flipout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
0sequential/dense_flipout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????z
0sequential/dense_flipout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
(sequential/dense_flipout_1/strided_sliceStridedSlice)sequential/dense_flipout_1/Shape:output:07sequential/dense_flipout_1/strided_slice/stack:output:09sequential/dense_flipout_1/strided_slice/stack_1:output:09sequential/dense_flipout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Fsequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Dsequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Dsequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
@sequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seedRandomUniformIntOsequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seed/shape:output:0Msequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seed/min:output:0Msequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Jsequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Jsequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
csequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterIsequential/dense_flipout_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Jsequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Fsequential/dense_flipout_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2)sequential/dense_flipout_1/Shape:output:0isequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0msequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Ssequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/alg:output:0Ssequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/min:output:0Ssequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	m
+sequential/dense_flipout_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
)sequential/dense_flipout_1/rademacher/mulMul4sequential/dense_flipout_1/rademacher/mul/x:output:0Osequential/dense_flipout_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????m
+sequential/dense_flipout_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
)sequential/dense_flipout_1/rademacher/subSub-sequential/dense_flipout_1/rademacher/mul:z:04sequential/dense_flipout_1/rademacher/sub/y:output:0*
T0	*(
_output_shapes
:???????????
*sequential/dense_flipout_1/rademacher/CastCast-sequential/dense_flipout_1/rademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????m
+sequential/dense_flipout_1/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :
k
)sequential/dense_flipout_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
%sequential/dense_flipout_1/ExpandDims
ExpandDims4sequential/dense_flipout_1/ExpandDims/input:output:02sequential/dense_flipout_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:h
&sequential/dense_flipout_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
!sequential/dense_flipout_1/concatConcatV21sequential/dense_flipout_1/strided_slice:output:0.sequential/dense_flipout_1/ExpandDims:output:0/sequential/dense_flipout_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Hsequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Fsequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
Fsequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
Bsequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntQsequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Osequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0Osequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Lsequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Lsequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
esequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterKsequential/dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Lsequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
Hsequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2*sequential/dense_flipout_1/concat:output:0ksequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0osequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Usequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Usequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Usequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????
*
dtype0	o
-sequential/dense_flipout_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
+sequential/dense_flipout_1/rademacher_1/mulMul6sequential/dense_flipout_1/rademacher_1/mul/x:output:0Qsequential/dense_flipout_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????
o
-sequential/dense_flipout_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
+sequential/dense_flipout_1/rademacher_1/subSub/sequential/dense_flipout_1/rademacher_1/mul:z:06sequential/dense_flipout_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
?
,sequential/dense_flipout_1/rademacher_1/CastCast/sequential/dense_flipout_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????
?
sequential/dense_flipout_1/mulMul+sequential/dense_flipout/Relu:activations:0.sequential/dense_flipout_1/rademacher/Cast:y:0*
T0*(
_output_shapes
:???????????
!sequential/dense_flipout_1/MatMulMatMul"sequential/dense_flipout_1/mul:z:09sequential/dense_flipout_1/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
 sequential/dense_flipout_1/mul_1Mul+sequential/dense_flipout_1/MatMul:product:00sequential/dense_flipout_1/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????
?
2sequential/dense_flipout_1/MatMul_1/ReadVariableOpReadVariableOp;sequential_dense_flipout_1_matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
#sequential/dense_flipout_1/MatMul_1MatMul+sequential/dense_flipout/Relu:activations:0:sequential/dense_flipout_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
sequential/dense_flipout_1/addAddV2-sequential/dense_flipout_1/MatMul_1:product:0$sequential/dense_flipout_1/mul_1:z:0*
T0*'
_output_shapes
:?????????
?
fsequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
{sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
}sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOpReadVariableOp?sequential_dense_flipout_1_independentdeterministic_constructed_at_dense_flipout_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:
*
dtype0?
~sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
?
tsequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
|sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_sliceStridedSlice?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/shape_as_tensor:output:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack:output:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_1:output:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
|sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgs?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
~sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
~sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
zsequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
usequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concatConcatV2?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_0:output:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs:r0:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_2:output:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
zsequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastToBroadcastTo?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp:value:0~sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:
?
|sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
vsequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReshapeReshape?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastTo:output:0?sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
?
gsequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
asequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/ReshapeReshapesequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape:output:0psequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
:
?
"sequential/dense_flipout_1/BiasAddBiasAdd"sequential/dense_flipout_1/add:z:0jsequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOpIsequential_dense_flipout_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?sequential_dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556894*
T0*
_output_shapes
: ?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp;sequential_dense_flipout_1_matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?sequential_dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556894*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?sequential_dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?sequential_dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2556894*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
,sequential/dense_flipout_1/divergence_kernelIdentity?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: z
IdentityIdentity+sequential/dense_flipout_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
?
NoOpNoOp?^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOpA^sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1.^sequential/batch_normalization/ReadVariableOp0^sequential/batch_normalization/ReadVariableOp_1A^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOpC^sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_10^sequential/batch_normalization_1/ReadVariableOp2^sequential/batch_normalization_1/ReadVariableOp_10^sequential/conv2d_flipout/Conv2D/ReadVariableOp|^sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp?^sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp@^sequential/conv2d_flipout/Normal/sample/Softplus/ReadVariableOp2^sequential/conv2d_flipout_1/Conv2D/ReadVariableOp?^sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp?^sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpB^sequential/conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOpz^sequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp?^sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp1^sequential/dense_flipout/MatMul_1/ReadVariableOp?^sequential/dense_flipout/Normal/sample/Softplus/ReadVariableOp~^sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp?^sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp3^sequential/dense_flipout_1/MatMul_1/ReadVariableOpA^sequential/dense_flipout_1/Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
2?
>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp>sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp2?
@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_1@sequential/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^
-sequential/batch_normalization/ReadVariableOp-sequential/batch_normalization/ReadVariableOp2b
/sequential/batch_normalization/ReadVariableOp_1/sequential/batch_normalization/ReadVariableOp_12?
@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp@sequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2?
Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1Bsequential/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12b
/sequential/batch_normalization_1/ReadVariableOp/sequential/batch_normalization_1/ReadVariableOp2f
1sequential/batch_normalization_1/ReadVariableOp_11sequential/batch_normalization_1/ReadVariableOp_12b
/sequential/conv2d_flipout/Conv2D/ReadVariableOp/sequential/conv2d_flipout/Conv2D/ReadVariableOp2?
{sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp{sequential/conv2d_flipout/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp2?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?sequential/conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2?
?sequential/conv2d_flipout/Normal/sample/Softplus/ReadVariableOp?sequential/conv2d_flipout/Normal/sample/Softplus/ReadVariableOp2f
1sequential/conv2d_flipout_1/Conv2D/ReadVariableOp1sequential/conv2d_flipout_1/Conv2D/ReadVariableOp2?
sequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOpsequential/conv2d_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp2?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?sequential/conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2?
Asequential/conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOpAsequential/conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp2?
ysequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOpysequential/dense_flipout/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp2?
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?sequential/dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2d
0sequential/dense_flipout/MatMul_1/ReadVariableOp0sequential/dense_flipout/MatMul_1/ReadVariableOp2?
>sequential/dense_flipout/Normal/sample/Softplus/ReadVariableOp>sequential/dense_flipout/Normal/sample/Softplus/ReadVariableOp2?
}sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp}sequential/dense_flipout_1/IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp2?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?sequential/dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2h
2sequential/dense_flipout_1/MatMul_1/ReadVariableOp2sequential/dense_flipout_1/MatMul_1/ReadVariableOp2?
@sequential/dense_flipout_1/Normal/sample/Softplus/ReadVariableOp@sequential/dense_flipout_1/Normal/sample/Softplus/ReadVariableOp:e a
/
_output_shapes
:?????????
.
_user_specified_nameconv2d_flipout_input:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2559949

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
,__inference_sequential_layer_call_fn_2557807
conv2d_flipout_input!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: #
	unknown_8: @#
	unknown_9: @

unknown_10:@

unknown_11

unknown_12

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
??

unknown_18:
??

unknown_19:	?

unknown_20

unknown_21

unknown_22:	?


unknown_23:	?


unknown_24:


unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_flipout_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????
: : : : *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2557744o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????
.
_user_specified_nameconv2d_flipout_input:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?	
?
2__inference_conv2d_flipout_1_layer_call_fn_2559993

inputs!
unknown: @#
	unknown_0: @
	unknown_1:@
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2557397w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:????????? : : : : : @22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: @
?	
?
1__inference_dense_flipout_1_layer_call_fn_2560405

inputs
unknown:	?

	unknown_0:	?

	unknown_1:

	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????
: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2557726o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':??????????: : : : :	?
22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :%!

_output_shapes
:	?

?	
?
7__inference_batch_normalization_1_layer_call_fn_2560163

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557007?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?>
?
G__inference_sequential_layer_call_and_return_conditional_losses_2558319
conv2d_flipout_input0
conv2d_flipout_2558245: 0
conv2d_flipout_2558247: $
conv2d_flipout_2558249: 
conv2d_flipout_2558251
conv2d_flipout_2558253)
batch_normalization_2558257: )
batch_normalization_2558259: )
batch_normalization_2558261: )
batch_normalization_2558263: 2
conv2d_flipout_1_2558267: @2
conv2d_flipout_1_2558269: @&
conv2d_flipout_1_2558271:@
conv2d_flipout_1_2558273
conv2d_flipout_1_2558275+
batch_normalization_1_2558279:@+
batch_normalization_1_2558281:@+
batch_normalization_1_2558283:@+
batch_normalization_1_2558285:@)
dense_flipout_2558290:
??)
dense_flipout_2558292:
??$
dense_flipout_2558294:	?
dense_flipout_2558296
dense_flipout_2558298*
dense_flipout_1_2558302:	?
*
dense_flipout_1_2558304:	?
%
dense_flipout_1_2558306:

dense_flipout_1_2558308
dense_flipout_1_2558310
identity

identity_1

identity_2

identity_3

identity_4??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?&conv2d_flipout/StatefulPartitionedCall?(conv2d_flipout_1/StatefulPartitionedCall?%dense_flipout/StatefulPartitionedCall?'dense_flipout_1/StatefulPartitionedCall?
&conv2d_flipout/StatefulPartitionedCallStatefulPartitionedCallconv2d_flipout_inputconv2d_flipout_2558245conv2d_flipout_2558247conv2d_flipout_2558249conv2d_flipout_2558251conv2d_flipout_2558253*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? : *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2557211?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall/conv2d_flipout/StatefulPartitionedCall:output:0batch_normalization_2558257batch_normalization_2558259batch_normalization_2558261batch_normalization_2558263*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556974?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2557238?
(conv2d_flipout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_flipout_1_2558267conv2d_flipout_1_2558269conv2d_flipout_1_2558271conv2d_flipout_1_2558273conv2d_flipout_1_2558275*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2557397?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_flipout_1/StatefulPartitionedCall:output:0batch_normalization_1_2558279batch_normalization_1_2558281batch_normalization_1_2558283batch_normalization_1_2558285*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557038?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2557424?
flatten/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2557432?
%dense_flipout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_flipout_2558290dense_flipout_2558292dense_flipout_2558294dense_flipout_2558296dense_flipout_2558298*
Tin

2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2557574?
'dense_flipout_1/StatefulPartitionedCallStatefulPartitionedCall.dense_flipout/StatefulPartitionedCall:output:0dense_flipout_1_2558302dense_flipout_1_2558304dense_flipout_1_2558306dense_flipout_1_2558308dense_flipout_1_2558310*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????
: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2557726
IdentityIdentity0dense_flipout_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
o

Identity_1Identity/conv2d_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: q

Identity_2Identity1conv2d_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: n

Identity_3Identity.dense_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p

Identity_4Identity0dense_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall'^conv2d_flipout/StatefulPartitionedCall)^conv2d_flipout_1/StatefulPartitionedCall&^dense_flipout/StatefulPartitionedCall(^dense_flipout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2P
&conv2d_flipout/StatefulPartitionedCall&conv2d_flipout/StatefulPartitionedCall2T
(conv2d_flipout_1/StatefulPartitionedCall(conv2d_flipout_1/StatefulPartitionedCall2N
%dense_flipout/StatefulPartitionedCall%dense_flipout/StatefulPartitionedCall2R
'dense_flipout_1/StatefulPartitionedCall'dense_flipout_1/StatefulPartitionedCall:e a
/
_output_shapes
:?????????
.
_user_specified_nameconv2d_flipout_input:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

??
?
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2557726

inputsA
.normal_sample_softplus_readvariableop_resource:	?
3
 matmul_1_readvariableop_resource:	?
y
kindependentdeterministic_constructed_at_dense_flipout_1_sample_deterministic_sample_readvariableop_resource:
?
?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557698?
?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?MatMul_1/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOpk
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes
:	?
]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0{
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	?
n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   
   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:?
*
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:?
?
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:?
?
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*#
_output_shapes
:?
v
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*#
_output_shapes
:?
l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	?
;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rw
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*(
_output_shapes
:??????????m
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????R
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????
*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????
T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????
Z
mulMulinputsrademacher/Cast:y:0*
T0*(
_output_shapes
:??????????k
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????
y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
?
KIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
qIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOpReadVariableOpkindependentdeterministic_constructed_at_dense_flipout_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:
*
dtype0?
cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
?
YIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
gIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
iIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
iIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_sliceStridedSlicelIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/shape_as_tensor:output:0pIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack:output:0rIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_1:output:0rIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
dIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
fIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgsoIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
ZIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concatConcatV2lIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_0:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs:r0:0lIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_2:output:0hIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastToBroadcastTojIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp:value:0cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:
?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
[IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReshapeReshapehIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastTo:output:0jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
?
LIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
FIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/ReshapeReshapedIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape:output:0UIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
:
?
BiasAddBiasAddadd:z:0OIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557698*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557698*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557698*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
zKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpc^IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':??????????: : : : :	?
2?
bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOpbIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :%!

_output_shapes
:	?

?
?
%__inference_signature_wrapper_2559734
conv2d_flipout_input!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: #
	unknown_8: @#
	unknown_9: @

unknown_10:@

unknown_11

unknown_12

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
??

unknown_18:
??

unknown_19:	?

unknown_20

unknown_21

unknown_22:	?


unknown_23:	?


unknown_24:


unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_flipout_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_2556921o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
22
StatefulPartitionedCallStatefulPartitionedCall:e a
/
_output_shapes
:?????????
.
_user_specified_nameconv2d_flipout_input:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
?
,__inference_sequential_layer_call_fn_2558384

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2
	unknown_3
	unknown_4: 
	unknown_5: 
	unknown_6: 
	unknown_7: #
	unknown_8: @#
	unknown_9: @

unknown_10:@

unknown_11

unknown_12

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:
??

unknown_18:
??

unknown_19:	?

unknown_20

unknown_21

unknown_22:	?


unknown_23:	?


unknown_24:


unknown_25

unknown_26
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26*(
Tin!
2*
Tout	
2*
_collective_manager_ids
 */
_output_shapes
:?????????
: : : : *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_2557744o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556974

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
G__inference_activation_layer_call_and_return_conditional_losses_2559977

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
??
?
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2557211

inputsH
.normal_sample_softplus_readvariableop_resource: 8
conv2d_readvariableop_resource: x
jindependentdeterministic_constructed_at_conv2d_flipout_sample_deterministic_sample_readvariableop_resource: ?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557183?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??Conv2D/ReadVariableOp?aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOpo

zeros_likeConst*&
_output_shapes
: *
dtype0*%
valueB *    ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: v
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskx
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: ?
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: ?
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0**
_output_shapes
: }
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0**
_output_shapes
: t
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: |
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsstrided_slice_1:output:0ExpandDims_1/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2ExpandDims:output:0ExpandDims_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:u
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????T
ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B : R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : w
ExpandDims_2
ExpandDimsExpandDims_2/input:output:0ExpandDims_2/dim:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2ExpandDims:output:0ExpandDims_2:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat_1:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? R
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_3
ExpandDimsrademacher/Cast:y:0ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:?????????R
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_4
ExpandDimsrademacher_1/Cast:y:0ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:????????? R
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_5
ExpandDimsExpandDims_3:output:0ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????R
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_6
ExpandDimsExpandDims_4:output:0ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:????????? c
mulMulinputsExpandDims_5:output:0*
T0*/
_output_shapes
:??????????
Conv2D_1Conv2Dmul:z:0Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
p
mul_1MulConv2D_1:output:0ExpandDims_6:output:0*
T0*/
_output_shapes
:????????? b
addAddV2Conv2D:output:0	mul_1:z:0*
T0*/
_output_shapes
:????????? ?
JIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
_IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
pIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOpReadVariableOpjindependentdeterministic_constructed_at_conv2d_flipout_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: ?
XIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
fIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
hIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
hIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_sliceStridedSlicekIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/shape_as_tensor:output:0oIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack:output:0qIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_1:output:0qIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
eIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgsBroadcastArgsnIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0iIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
YIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concatConcatV2kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_0:output:0eIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs:r0:0kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_2:output:0gIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastToBroadcastToiIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp:value:0bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: ?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ?
ZIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReshapeReshapegIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastTo:output:0iIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ?
KIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
EIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/ReshapeReshapecIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape:output:0TIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape/shape:output:0*
T0*
_output_shapes
: ?
BiasAddBiasAddadd:z:0NIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557183*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557183*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557183*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
yKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^Conv2D/ReadVariableOpb^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:?????????: : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOpaIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: 
?>
?
G__inference_sequential_layer_call_and_return_conditional_losses_2558242
conv2d_flipout_input0
conv2d_flipout_2558168: 0
conv2d_flipout_2558170: $
conv2d_flipout_2558172: 
conv2d_flipout_2558174
conv2d_flipout_2558176)
batch_normalization_2558180: )
batch_normalization_2558182: )
batch_normalization_2558184: )
batch_normalization_2558186: 2
conv2d_flipout_1_2558190: @2
conv2d_flipout_1_2558192: @&
conv2d_flipout_1_2558194:@
conv2d_flipout_1_2558196
conv2d_flipout_1_2558198+
batch_normalization_1_2558202:@+
batch_normalization_1_2558204:@+
batch_normalization_1_2558206:@+
batch_normalization_1_2558208:@)
dense_flipout_2558213:
??)
dense_flipout_2558215:
??$
dense_flipout_2558217:	?
dense_flipout_2558219
dense_flipout_2558221*
dense_flipout_1_2558225:	?
*
dense_flipout_1_2558227:	?
%
dense_flipout_1_2558229:

dense_flipout_1_2558231
dense_flipout_1_2558233
identity

identity_1

identity_2

identity_3

identity_4??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?&conv2d_flipout/StatefulPartitionedCall?(conv2d_flipout_1/StatefulPartitionedCall?%dense_flipout/StatefulPartitionedCall?'dense_flipout_1/StatefulPartitionedCall?
&conv2d_flipout/StatefulPartitionedCallStatefulPartitionedCallconv2d_flipout_inputconv2d_flipout_2558168conv2d_flipout_2558170conv2d_flipout_2558172conv2d_flipout_2558174conv2d_flipout_2558176*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? : *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2557211?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall/conv2d_flipout/StatefulPartitionedCall:output:0batch_normalization_2558180batch_normalization_2558182batch_normalization_2558184batch_normalization_2558186*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556943?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2557238?
(conv2d_flipout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_flipout_1_2558190conv2d_flipout_1_2558192conv2d_flipout_1_2558194conv2d_flipout_1_2558196conv2d_flipout_1_2558198*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2557397?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_flipout_1/StatefulPartitionedCall:output:0batch_normalization_1_2558202batch_normalization_1_2558204batch_normalization_1_2558206batch_normalization_1_2558208*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557007?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2557424?
flatten/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2557432?
%dense_flipout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_flipout_2558213dense_flipout_2558215dense_flipout_2558217dense_flipout_2558219dense_flipout_2558221*
Tin

2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2557574?
'dense_flipout_1/StatefulPartitionedCallStatefulPartitionedCall.dense_flipout/StatefulPartitionedCall:output:0dense_flipout_1_2558225dense_flipout_1_2558227dense_flipout_1_2558229dense_flipout_1_2558231dense_flipout_1_2558233*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????
: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2557726
IdentityIdentity0dense_flipout_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
o

Identity_1Identity/conv2d_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: q

Identity_2Identity1conv2d_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: n

Identity_3Identity.dense_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p

Identity_4Identity0dense_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall'^conv2d_flipout/StatefulPartitionedCall)^conv2d_flipout_1/StatefulPartitionedCall&^dense_flipout/StatefulPartitionedCall(^dense_flipout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2P
&conv2d_flipout/StatefulPartitionedCall&conv2d_flipout/StatefulPartitionedCall2T
(conv2d_flipout_1/StatefulPartitionedCall(conv2d_flipout_1/StatefulPartitionedCall2N
%dense_flipout/StatefulPartitionedCall%dense_flipout/StatefulPartitionedCall2R
'dense_flipout_1/StatefulPartitionedCall'dense_flipout_1/StatefulPartitionedCall:e a
/
_output_shapes
:?????????
.
_user_specified_nameconv2d_flipout_input:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
e
I__inference_activation_1_layer_call_and_return_conditional_losses_2560222

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:?????????@b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2559967

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556943

inputs%
readvariableop_resource: '
readvariableop_1_resource: 6
(fusedbatchnormv3_readvariableop_resource: 8
*fusedbatchnormv3_readvariableop_1_resource: 
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? ?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?	
?
5__inference_batch_normalization_layer_call_fn_2559918

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556943?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2560544

inputsA
.normal_sample_softplus_readvariableop_resource:	?
3
 matmul_1_readvariableop_resource:	?
y
kindependentdeterministic_constructed_at_dense_flipout_1_sample_deterministic_sample_readvariableop_resource:
?
?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560516?
?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?MatMul_1/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOpk
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    |

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*
_output_shapes
:	?
]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0{
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	?
n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   
   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:?
*
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:?
?
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:?
?
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*#
_output_shapes
:?
v
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*#
_output_shapes
:?
l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	?
;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rw
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*(
_output_shapes
:??????????m
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????R
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :
P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????
*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????
T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????
Z
mulMulinputsrademacher/Cast:y:0*
T0*(
_output_shapes
:??????????k
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
g
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????
y
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0m
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
]
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*'
_output_shapes
:?????????
?
KIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
qIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOpReadVariableOpkindependentdeterministic_constructed_at_dense_flipout_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:
*
dtype0?
cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
?
YIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
gIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
iIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
iIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_sliceStridedSlicelIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/shape_as_tensor:output:0pIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack:output:0rIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_1:output:0rIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
dIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
fIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgsoIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
ZIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concatConcatV2lIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_0:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastArgs:r0:0lIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/values_2:output:0hIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastToBroadcastTojIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp:value:0cIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:
?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
[IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReshapeReshapehIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/BroadcastTo:output:0jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
?
LIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
FIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/ReshapeReshapedIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/Reshape:output:0UIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
:
?
BiasAddBiasAddadd:z:0OIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560516*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560516*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560516*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
zKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpc^IndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':??????????: : : : :	?
2?
bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOpbIndependentDeterministic_CONSTRUCTED_AT_dense_flipout_1/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :%!

_output_shapes
:	?

?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557038

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??
?
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2557397

inputsH
.normal_sample_softplus_readvariableop_resource: @8
conv2d_readvariableop_resource: @z
lindependentdeterministic_constructed_at_conv2d_flipout_1_sample_deterministic_sample_readvariableop_resource:@?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557369?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??Conv2D/ReadVariableOp?cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOps
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0*&
_output_shapes
: @]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: @v
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskx
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"          @   W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0**
_output_shapes
: @*
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: @?
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: @?
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0**
_output_shapes
: @}
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0**
_output_shapes
: @t
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: @|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsstrided_slice_1:output:0ExpandDims_1/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2ExpandDims:output:0ExpandDims_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:u
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? T
ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B :@R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : w
ExpandDims_2
ExpandDimsExpandDims_2/input:output:0ExpandDims_2/dim:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2ExpandDims:output:0ExpandDims_2:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat_1:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????@*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????@T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????@p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????@R
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_3
ExpandDimsrademacher/Cast:y:0ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:????????? R
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_4
ExpandDimsrademacher_1/Cast:y:0ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:?????????@R
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_5
ExpandDimsExpandDims_3:output:0ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:????????? R
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_6
ExpandDimsExpandDims_4:output:0ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:?????????@c
mulMulinputsExpandDims_5:output:0*
T0*/
_output_shapes
:????????? ?
Conv2D_1Conv2Dmul:z:0Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
p
mul_1MulConv2D_1:output:0ExpandDims_6:output:0*
T0*/
_output_shapes
:?????????@b
addAddV2Conv2D:output:0	mul_1:z:0*
T0*/
_output_shapes
:?????????@?
LIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
rIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOpReadVariableOplindependentdeterministic_constructed_at_conv2d_flipout_1_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:@*
dtype0?
dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:@?
ZIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
hIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
jIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
jIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_sliceStridedSlicemIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/shape_as_tensor:output:0qIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack:output:0sIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_1:output:0sIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
eIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
gIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgsBroadcastArgspIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
[IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concatConcatV2mIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_0:output:0gIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastArgs:r0:0mIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/values_2:output:0iIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastToBroadcastTokIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp:value:0dIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:@?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
\IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReshapeReshapeiIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/BroadcastTo:output:0kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:@?
MIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:@?
GIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/ReshapeReshapeeIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/Reshape:output:0VIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape/shape:output:0*
T0*
_output_shapes
:@?
BiasAddBiasAddadd:z:0PIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557369*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557369*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557369*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
{KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:?????????@Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^Conv2D/ReadVariableOpd^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:????????? : : : : : @2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOpcIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout_1/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: @
?=
?
G__inference_sequential_layer_call_and_return_conditional_losses_2558037

inputs0
conv2d_flipout_2557963: 0
conv2d_flipout_2557965: $
conv2d_flipout_2557967: 
conv2d_flipout_2557969
conv2d_flipout_2557971)
batch_normalization_2557975: )
batch_normalization_2557977: )
batch_normalization_2557979: )
batch_normalization_2557981: 2
conv2d_flipout_1_2557985: @2
conv2d_flipout_1_2557987: @&
conv2d_flipout_1_2557989:@
conv2d_flipout_1_2557991
conv2d_flipout_1_2557993+
batch_normalization_1_2557997:@+
batch_normalization_1_2557999:@+
batch_normalization_1_2558001:@+
batch_normalization_1_2558003:@)
dense_flipout_2558008:
??)
dense_flipout_2558010:
??$
dense_flipout_2558012:	?
dense_flipout_2558014
dense_flipout_2558016*
dense_flipout_1_2558020:	?
*
dense_flipout_1_2558022:	?
%
dense_flipout_1_2558024:

dense_flipout_1_2558026
dense_flipout_1_2558028
identity

identity_1

identity_2

identity_3

identity_4??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?&conv2d_flipout/StatefulPartitionedCall?(conv2d_flipout_1/StatefulPartitionedCall?%dense_flipout/StatefulPartitionedCall?'dense_flipout_1/StatefulPartitionedCall?
&conv2d_flipout/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_flipout_2557963conv2d_flipout_2557965conv2d_flipout_2557967conv2d_flipout_2557969conv2d_flipout_2557971*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? : *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2557211?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall/conv2d_flipout/StatefulPartitionedCall:output:0batch_normalization_2557975batch_normalization_2557977batch_normalization_2557979batch_normalization_2557981*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556974?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2557238?
(conv2d_flipout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_flipout_1_2557985conv2d_flipout_1_2557987conv2d_flipout_1_2557989conv2d_flipout_1_2557991conv2d_flipout_1_2557993*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2557397?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_flipout_1/StatefulPartitionedCall:output:0batch_normalization_1_2557997batch_normalization_1_2557999batch_normalization_1_2558001batch_normalization_1_2558003*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557038?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2557424?
flatten/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2557432?
%dense_flipout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_flipout_2558008dense_flipout_2558010dense_flipout_2558012dense_flipout_2558014dense_flipout_2558016*
Tin

2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2557574?
'dense_flipout_1/StatefulPartitionedCallStatefulPartitionedCall.dense_flipout/StatefulPartitionedCall:output:0dense_flipout_1_2558020dense_flipout_1_2558022dense_flipout_1_2558024dense_flipout_1_2558026dense_flipout_1_2558028*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????
: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2557726
IdentityIdentity0dense_flipout_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
o

Identity_1Identity/conv2d_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: q

Identity_2Identity1conv2d_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: n

Identity_3Identity.dense_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p

Identity_4Identity0dense_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall'^conv2d_flipout/StatefulPartitionedCall)^conv2d_flipout_1/StatefulPartitionedCall&^dense_flipout/StatefulPartitionedCall(^dense_flipout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2P
&conv2d_flipout/StatefulPartitionedCall&conv2d_flipout/StatefulPartitionedCall2T
(conv2d_flipout_1/StatefulPartitionedCall(conv2d_flipout_1/StatefulPartitionedCall2N
%dense_flipout/StatefulPartitionedCall%dense_flipout/StatefulPartitionedCall2R
'dense_flipout_1/StatefulPartitionedCall'dense_flipout_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?	
?
0__inference_conv2d_flipout_layer_call_fn_2559750

inputs!
unknown: #
	unknown_0: 
	unknown_1: 
	unknown_2
	unknown_3
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? : *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2557211w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:?????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: 
?
J
.__inference_activation_1_layer_call_fn_2560217

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2557424h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2557574

inputsB
.normal_sample_softplus_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??x
iindependentdeterministic_constructed_at_dense_flipout_sample_deterministic_sample_readvariableop_resource:	??
?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557546?
?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?MatMul_1/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOpk
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    }

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0* 
_output_shapes
:
??]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0* 
_output_shapes
:
??n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"@     W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*$
_output_shapes
:??*
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*$
_output_shapes
:???
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*$
_output_shapes
:???
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*$
_output_shapes
:??w
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*$
_output_shapes
:??l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0* 
_output_shapes
:
??;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rw
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*(
_output_shapes
:??????????m
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????S
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value
B :?P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R}
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*(
_output_shapes
:??????????q
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????Z
mulMulinputsrademacher/Cast:y:0*
T0*(
_output_shapes
:??????????l
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*(
_output_shapes
:??????????h
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*(
_output_shapes
:???????????
IIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
^IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
oIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOpReadVariableOpiindependentdeterministic_constructed_at_dense_flipout_sample_deterministic_sample_readvariableop_resource*
_output_shapes	
:?*
dtype0?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:??
WIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
eIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
gIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
gIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_sliceStridedSlicejIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/shape_as_tensor:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack:output:0pIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_1:output:0pIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
dIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgsBroadcastArgsmIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0hIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
]IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
XIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concatConcatV2jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_0:output:0dIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs:r0:0jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_2:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
]IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastToBroadcastTohIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp:value:0aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes
:	??
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
YIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReshapeReshapefIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastTo:output:0hIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	??
JIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
DIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/ReshapeReshapebIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape:output:0SIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape/shape:output:0*
T0*
_output_shapes	
:??
BiasAddBiasAddadd:z:0MIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape:output:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557546*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557546*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2557546*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
xKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpa^IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : :
??2?
`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :&"
 
_output_shapes
:
??
?
c
G__inference_activation_layer_call_and_return_conditional_losses_2557238

inputs
identityN
ReluReluinputs*
T0*/
_output_shapes
:????????? b
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:????????? "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:????????? :W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?8
?
 __inference__traced_save_2560635
file_prefixB
>savev2_conv2d_flipout_kernel_posterior_loc_read_readvariableopR
Nsavev2_conv2d_flipout_kernel_posterior_untransformed_scale_read_readvariableop@
<savev2_conv2d_flipout_bias_posterior_loc_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableopD
@savev2_conv2d_flipout_1_kernel_posterior_loc_read_readvariableopT
Psavev2_conv2d_flipout_1_kernel_posterior_untransformed_scale_read_readvariableopB
>savev2_conv2d_flipout_1_bias_posterior_loc_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableopA
=savev2_dense_flipout_kernel_posterior_loc_read_readvariableopQ
Msavev2_dense_flipout_kernel_posterior_untransformed_scale_read_readvariableop?
;savev2_dense_flipout_bias_posterior_loc_read_readvariableopC
?savev2_dense_flipout_1_kernel_posterior_loc_read_readvariableopS
Osavev2_dense_flipout_1_kernel_posterior_untransformed_scale_read_readvariableopA
=savev2_dense_flipout_1_bias_posterior_loc_read_readvariableop
savev2_const_8

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?
B?
BDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-2/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-2/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-4/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-4/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-5/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-5/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B ?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0>savev2_conv2d_flipout_kernel_posterior_loc_read_readvariableopNsavev2_conv2d_flipout_kernel_posterior_untransformed_scale_read_readvariableop<savev2_conv2d_flipout_bias_posterior_loc_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop@savev2_conv2d_flipout_1_kernel_posterior_loc_read_readvariableopPsavev2_conv2d_flipout_1_kernel_posterior_untransformed_scale_read_readvariableop>savev2_conv2d_flipout_1_bias_posterior_loc_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop=savev2_dense_flipout_kernel_posterior_loc_read_readvariableopMsavev2_dense_flipout_kernel_posterior_untransformed_scale_read_readvariableop;savev2_dense_flipout_bias_posterior_loc_read_readvariableop?savev2_dense_flipout_1_kernel_posterior_loc_read_readvariableopOsavev2_dense_flipout_1_kernel_posterior_untransformed_scale_read_readvariableop=savev2_dense_flipout_1_bias_posterior_loc_read_readvariableopsavev2_const_8"/device:CPU:0*
_output_shapes
 *#
dtypes
2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : @: @:@:@:@:@:@:
??:
??:?:	?
:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @:,	(
&
_output_shapes
: @: 


_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:&"
 
_output_shapes
:
??:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?
:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: 
?Y
?
#__inference__traced_restore_2560705
file_prefixN
4assignvariableop_conv2d_flipout_kernel_posterior_loc: `
Fassignvariableop_1_conv2d_flipout_kernel_posterior_untransformed_scale: B
4assignvariableop_2_conv2d_flipout_bias_posterior_loc: :
,assignvariableop_3_batch_normalization_gamma: 9
+assignvariableop_4_batch_normalization_beta: @
2assignvariableop_5_batch_normalization_moving_mean: D
6assignvariableop_6_batch_normalization_moving_variance: R
8assignvariableop_7_conv2d_flipout_1_kernel_posterior_loc: @b
Hassignvariableop_8_conv2d_flipout_1_kernel_posterior_untransformed_scale: @D
6assignvariableop_9_conv2d_flipout_1_bias_posterior_loc:@=
/assignvariableop_10_batch_normalization_1_gamma:@<
.assignvariableop_11_batch_normalization_1_beta:@C
5assignvariableop_12_batch_normalization_1_moving_mean:@G
9assignvariableop_13_batch_normalization_1_moving_variance:@J
6assignvariableop_14_dense_flipout_kernel_posterior_loc:
??Z
Fassignvariableop_15_dense_flipout_kernel_posterior_untransformed_scale:
??C
4assignvariableop_16_dense_flipout_bias_posterior_loc:	?K
8assignvariableop_17_dense_flipout_1_kernel_posterior_loc:	?
[
Hassignvariableop_18_dense_flipout_1_kernel_posterior_untransformed_scale:	?
D
6assignvariableop_19_dense_flipout_1_bias_posterior_loc:

identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?
B?
BDlayer_with_weights-0/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-2/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-2/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-4/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-4/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBDlayer_with_weights-5/kernel_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-5/kernel_posterior_untransformed_scale/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-5/bias_posterior_loc/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp4assignvariableop_conv2d_flipout_kernel_posterior_locIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOpFassignvariableop_1_conv2d_flipout_kernel_posterior_untransformed_scaleIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp4assignvariableop_2_conv2d_flipout_bias_posterior_locIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batch_normalization_gammaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_batch_normalization_betaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp2assignvariableop_5_batch_normalization_moving_meanIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp6assignvariableop_6_batch_normalization_moving_varianceIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp8assignvariableop_7_conv2d_flipout_1_kernel_posterior_locIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOpHassignvariableop_8_conv2d_flipout_1_kernel_posterior_untransformed_scaleIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_conv2d_flipout_1_bias_posterior_locIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_dense_flipout_kernel_posterior_locIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOpFassignvariableop_15_dense_flipout_kernel_posterior_untransformed_scaleIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp4assignvariableop_16_dense_flipout_bias_posterior_locIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp8assignvariableop_17_dense_flipout_1_kernel_posterior_locIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOpHassignvariableop_18_dense_flipout_1_kernel_posterior_untransformed_scaleIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_dense_flipout_1_bias_posterior_locIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_2557432

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_2560233

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2559905

inputsH
.normal_sample_softplus_readvariableop_resource: 8
conv2d_readvariableop_resource: x
jindependentdeterministic_constructed_at_conv2d_flipout_sample_deterministic_sample_readvariableop_resource: ?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559877?
?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??Conv2D/ReadVariableOp?aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOpo

zeros_likeConst*&
_output_shapes
: *
dtype0*%
valueB *    ]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: v
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskx
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: ?
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: ?
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0**
_output_shapes
: }
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0**
_output_shapes
: t
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: |
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : n

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:h
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskR
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : t
ExpandDims_1
ExpandDimsstrided_slice_1:output:0ExpandDims_1/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2ExpandDims:output:0ExpandDims_1:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:u
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rv
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*'
_output_shapes
:?????????l
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????T
ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B : R
ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : w
ExpandDims_2
ExpandDimsExpandDims_2/input:output:0ExpandDims_2/dim:output:0*
T0*
_output_shapes
:O
concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concat_1ConcatV2ExpandDims:output:0ExpandDims_2:output:0concat_1/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat_1:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R|
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? p
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? R
ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_3
ExpandDimsrademacher/Cast:y:0ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:?????????R
ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_4
ExpandDimsrademacher_1/Cast:y:0ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:????????? R
ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_5
ExpandDimsExpandDims_3:output:0ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????R
ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
ExpandDims_6
ExpandDimsExpandDims_4:output:0ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:????????? c
mulMulinputsExpandDims_5:output:0*
T0*/
_output_shapes
:??????????
Conv2D_1Conv2Dmul:z:0Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
p
mul_1MulConv2D_1:output:0ExpandDims_6:output:0*
T0*/
_output_shapes
:????????? b
addAddV2Conv2D:output:0	mul_1:z:0*
T0*/
_output_shapes
:????????? ?
JIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
_IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
pIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOpReadVariableOpjindependentdeterministic_constructed_at_conv2d_flipout_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: ?
XIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
fIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
hIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
hIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_sliceStridedSlicekIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/shape_as_tensor:output:0oIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack:output:0qIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_1:output:0qIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
cIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
eIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgsBroadcastArgsnIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0iIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
YIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concatConcatV2kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_0:output:0eIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastArgs:r0:0kIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/values_2:output:0gIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastToBroadcastToiIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp:value:0bIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: ?
`IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ?
ZIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReshapeReshapegIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/BroadcastTo:output:0iIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ?
KIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
EIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/ReshapeReshapecIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/Reshape:output:0TIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape/shape:output:0*
T0*
_output_shapes
: ?
BiasAddBiasAddadd:z:0NIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559877*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559877*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559877*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
yKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: g
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:????????? Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp^Conv2D/ReadVariableOpb^IndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:?????????: : : : : 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2?
aIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOpaIndependentDeterministic_CONSTRUCTED_AT_conv2d_flipout/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: 
?	
?
7__inference_batch_normalization_1_layer_call_fn_2560176

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557038?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
??	
?4
G__inference_sequential_layer_call_and_return_conditional_losses_2559060

inputsW
=conv2d_flipout_normal_sample_softplus_readvariableop_resource: G
-conv2d_flipout_conv2d_readvariableop_resource: i
[conv2d_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource: ?
?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558576?
?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x9
+batch_normalization_readvariableop_resource: ;
-batch_normalization_readvariableop_1_resource: J
<batch_normalization_fusedbatchnormv3_readvariableop_resource: L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource: Y
?conv2d_flipout_1_normal_sample_softplus_readvariableop_resource: @I
/conv2d_flipout_1_conv2d_readvariableop_resource: @k
]conv2d_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource:@?
?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558743?
?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x;
-batch_normalization_1_readvariableop_resource:@=
/batch_normalization_1_readvariableop_1_resource:@L
>batch_normalization_1_fusedbatchnormv3_readvariableop_resource:@N
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:@P
<dense_flipout_normal_sample_softplus_readvariableop_resource:
??B
.dense_flipout_matmul_1_readvariableop_resource:
??i
Zdense_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource:	??
?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558895?
?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_xQ
>dense_flipout_1_normal_sample_softplus_readvariableop_resource:	?
C
0dense_flipout_1_matmul_1_readvariableop_resource:	?
j
\dense_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource:
?
?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559029?
?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1

identity_2

identity_3

identity_4??3batch_normalization/FusedBatchNormV3/ReadVariableOp?5batch_normalization/FusedBatchNormV3/ReadVariableOp_1?"batch_normalization/ReadVariableOp?$batch_normalization/ReadVariableOp_1?5batch_normalization_1/FusedBatchNormV3/ReadVariableOp?7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1?$batch_normalization_1/ReadVariableOp?&batch_normalization_1/ReadVariableOp_1?$conv2d_flipout/Conv2D/ReadVariableOp?Rconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?4conv2d_flipout/Normal/sample/Softplus/ReadVariableOp?&conv2d_flipout_1/Conv2D/ReadVariableOp?Tconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp?Qdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?%dense_flipout/MatMul_1/ReadVariableOp?3dense_flipout/Normal/sample/Softplus/ReadVariableOp?Sdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp??dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?'dense_flipout_1/MatMul_1/ReadVariableOp?5dense_flipout_1/Normal/sample/Softplus/ReadVariableOp~
conv2d_flipout/zeros_likeConst*&
_output_shapes
: *
dtype0*%
valueB *    l
)conv2d_flipout/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
4conv2d_flipout/Normal/sample/Softplus/ReadVariableOpReadVariableOp=conv2d_flipout_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
%conv2d_flipout/Normal/sample/SoftplusSoftplus<conv2d_flipout/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: g
"conv2d_flipout/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
 conv2d_flipout/Normal/sample/addAddV2+conv2d_flipout/Normal/sample/add/x:output:03conv2d_flipout/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: ?
,conv2d_flipout/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"             d
"conv2d_flipout/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : z
0conv2d_flipout/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: |
2conv2d_flipout/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2conv2d_flipout/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
*conv2d_flipout/Normal/sample/strided_sliceStridedSlice5conv2d_flipout/Normal/sample/shape_as_tensor:output:09conv2d_flipout/Normal/sample/strided_slice/stack:output:0;conv2d_flipout/Normal/sample/strided_slice/stack_1:output:0;conv2d_flipout/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
.conv2d_flipout/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"             f
$conv2d_flipout/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : |
2conv2d_flipout/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4conv2d_flipout/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4conv2d_flipout/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,conv2d_flipout/Normal/sample/strided_slice_1StridedSlice7conv2d_flipout/Normal/sample/shape_as_tensor_1:output:0;conv2d_flipout/Normal/sample/strided_slice_1/stack:output:0=conv2d_flipout/Normal/sample/strided_slice_1/stack_1:output:0=conv2d_flipout/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
-conv2d_flipout/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB r
/conv2d_flipout/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
*conv2d_flipout/Normal/sample/BroadcastArgsBroadcastArgs8conv2d_flipout/Normal/sample/BroadcastArgs/s0_1:output:03conv2d_flipout/Normal/sample/strided_slice:output:0*
_output_shapes
:?
,conv2d_flipout/Normal/sample/BroadcastArgs_1BroadcastArgs/conv2d_flipout/Normal/sample/BroadcastArgs:r0:05conv2d_flipout/Normal/sample/strided_slice_1:output:0*
_output_shapes
:v
,conv2d_flipout/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:j
(conv2d_flipout/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
#conv2d_flipout/Normal/sample/concatConcatV25conv2d_flipout/Normal/sample/concat/values_0:output:01conv2d_flipout/Normal/sample/BroadcastArgs_1:r0:01conv2d_flipout/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:{
6conv2d_flipout/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    }
8conv2d_flipout/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Fconv2d_flipout/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal,conv2d_flipout/Normal/sample/concat:output:0*
T0**
_output_shapes
: *
dtype0?
5conv2d_flipout/Normal/sample/normal/random_normal/mulMulOconv2d_flipout/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Aconv2d_flipout/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: ?
1conv2d_flipout/Normal/sample/normal/random_normalAddV29conv2d_flipout/Normal/sample/normal/random_normal/mul:z:0?conv2d_flipout/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: ?
 conv2d_flipout/Normal/sample/mulMul5conv2d_flipout/Normal/sample/normal/random_normal:z:0$conv2d_flipout/Normal/sample/add:z:0*
T0**
_output_shapes
: ?
"conv2d_flipout/Normal/sample/add_1AddV2$conv2d_flipout/Normal/sample/mul:z:0"conv2d_flipout/zeros_like:output:0*
T0**
_output_shapes
: ?
*conv2d_flipout/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"             ?
$conv2d_flipout/Normal/sample/ReshapeReshape&conv2d_flipout/Normal/sample/add_1:z:03conv2d_flipout/Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: ?
$conv2d_flipout/Conv2D/ReadVariableOpReadVariableOp-conv2d_flipout_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
conv2d_flipout/Conv2DConv2Dinputs,conv2d_flipout/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
J
conv2d_flipout/ShapeShapeinputs*
T0*
_output_shapes
:l
"conv2d_flipout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: n
$conv2d_flipout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:n
$conv2d_flipout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_flipout/strided_sliceStridedSliceconv2d_flipout/Shape:output:0+conv2d_flipout/strided_slice/stack:output:0-conv2d_flipout/strided_slice/stack_1:output:0-conv2d_flipout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
conv2d_flipout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/ExpandDims
ExpandDims%conv2d_flipout/strided_slice:output:0&conv2d_flipout/ExpandDims/dim:output:0*
T0*
_output_shapes
:w
$conv2d_flipout/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????p
&conv2d_flipout/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_flipout/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_flipout/strided_slice_1StridedSliceconv2d_flipout/Shape:output:0-conv2d_flipout/strided_slice_1/stack:output:0/conv2d_flipout/strided_slice_1/stack_1:output:0/conv2d_flipout/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
conv2d_flipout/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/ExpandDims_1
ExpandDims'conv2d_flipout/strided_slice_1:output:0(conv2d_flipout/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:\
conv2d_flipout/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/concatConcatV2"conv2d_flipout/ExpandDims:output:0$conv2d_flipout/ExpandDims_1:output:0#conv2d_flipout/concat/axis:output:0*
N*
T0*
_output_shapes
:?
:conv2d_flipout/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
8conv2d_flipout/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????~
8conv2d_flipout/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
4conv2d_flipout/rademacher/uniform/sanitize_seed/seedRandomUniformIntCconv2d_flipout/rademacher/uniform/sanitize_seed/seed/shape:output:0Aconv2d_flipout/rademacher/uniform/sanitize_seed/seed/min:output:0Aconv2d_flipout/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
>conv2d_flipout/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
>conv2d_flipout/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Wconv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter=conv2d_flipout/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
>conv2d_flipout/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
:conv2d_flipout/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2conv2d_flipout/concat:output:0]conv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0aconv2d_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Gconv2d_flipout/rademacher/uniform/stateless_random_uniform/alg:output:0Gconv2d_flipout/rademacher/uniform/stateless_random_uniform/min:output:0Gconv2d_flipout/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????*
dtype0	a
conv2d_flipout/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher/mulMul(conv2d_flipout/rademacher/mul/x:output:0Cconv2d_flipout/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????a
conv2d_flipout/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher/subSub!conv2d_flipout/rademacher/mul:z:0(conv2d_flipout/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:??????????
conv2d_flipout/rademacher/CastCast!conv2d_flipout/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????c
!conv2d_flipout/ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B : a
conv2d_flipout/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/ExpandDims_2
ExpandDims*conv2d_flipout/ExpandDims_2/input:output:0(conv2d_flipout/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:^
conv2d_flipout/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout/concat_1ConcatV2"conv2d_flipout/ExpandDims:output:0$conv2d_flipout/ExpandDims_2:output:0%conv2d_flipout/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
<conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
:conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
6conv2d_flipout/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntEconv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Cconv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/min:output:0Cconv2d_flipout/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
@conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
@conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Yconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter?conv2d_flipout/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
@conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
<conv2d_flipout/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2 conv2d_flipout/concat_1:output:0_conv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0cconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Iconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/alg:output:0Iconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/min:output:0Iconv2d_flipout/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	c
!conv2d_flipout/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher_1/mulMul*conv2d_flipout/rademacher_1/mul/x:output:0Econv2d_flipout/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? c
!conv2d_flipout/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout/rademacher_1/subSub#conv2d_flipout/rademacher_1/mul:z:0*conv2d_flipout/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:????????? ?
 conv2d_flipout/rademacher_1/CastCast#conv2d_flipout/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? a
conv2d_flipout/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_3
ExpandDims"conv2d_flipout/rademacher/Cast:y:0(conv2d_flipout/ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:?????????a
conv2d_flipout/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_4
ExpandDims$conv2d_flipout/rademacher_1/Cast:y:0(conv2d_flipout/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:????????? a
conv2d_flipout/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_5
ExpandDims$conv2d_flipout/ExpandDims_3:output:0(conv2d_flipout/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:?????????a
conv2d_flipout/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout/ExpandDims_6
ExpandDims$conv2d_flipout/ExpandDims_4:output:0(conv2d_flipout/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_flipout/mulMulinputs$conv2d_flipout/ExpandDims_5:output:0*
T0*/
_output_shapes
:??????????
conv2d_flipout/Conv2D_1Conv2Dconv2d_flipout/mul:z:0-conv2d_flipout/Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
?
conv2d_flipout/mul_1Mul conv2d_flipout/Conv2D_1:output:0$conv2d_flipout/ExpandDims_6:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_flipout/addAddV2conv2d_flipout/Conv2D:output:0conv2d_flipout/mul_1:z:0*
T0*/
_output_shapes
:????????? ~
;conv2d_flipout/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Pconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
aconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Rconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOp[conv2d_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
: *
dtype0?
Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB: ?
Iconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Wconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Yconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Yconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Qconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice\conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0`conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0bconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0bconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Tconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Vconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Qconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs_conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Zconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Oconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Jconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2\conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Vconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0\conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Xconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Oconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToZconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Sconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

: ?
Qconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"       ?
Kconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeXconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Zconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

: ?
<conv2d_flipout/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: ?
6conv2d_flipout/IndependentDeterministic/sample/ReshapeReshapeTconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Econv2d_flipout/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
: ?
conv2d_flipout/BiasAddBiasAddconv2d_flipout/add:z:0?conv2d_flipout/IndependentDeterministic/sample/Reshape:output:0*
T0*/
_output_shapes
:????????? ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp=conv2d_flipout_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: *
dtype0?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558576*
T0*
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp-conv2d_flipout_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558576*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?conv2d_flipout_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558576*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: ?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
 conv2d_flipout/divergence_kernelIdentity?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: ?
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype0?
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype0?
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype0?
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype0?
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_flipout/BiasAdd:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:????????? : : : : :*
epsilon%o?:*
is_training( {
activation/ReluRelu(batch_normalization/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:????????? ?
+conv2d_flipout_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   f
!conv2d_flipout_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
conv2d_flipout_1/zeros_likeFill4conv2d_flipout_1/zeros_like/shape_as_tensor:output:0*conv2d_flipout_1/zeros_like/Const:output:0*
T0*&
_output_shapes
: @n
+conv2d_flipout_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp?conv2d_flipout_1_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
'conv2d_flipout_1/Normal/sample/SoftplusSoftplus>conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @i
$conv2d_flipout_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
"conv2d_flipout_1/Normal/sample/addAddV2-conv2d_flipout_1/Normal/sample/add/x:output:05conv2d_flipout_1/Normal/sample/Softplus:activations:0*
T0*&
_output_shapes
: @?
.conv2d_flipout_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*%
valueB"          @   f
$conv2d_flipout_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : |
2conv2d_flipout_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ~
4conv2d_flipout_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:~
4conv2d_flipout_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
,conv2d_flipout_1/Normal/sample/strided_sliceStridedSlice7conv2d_flipout_1/Normal/sample/shape_as_tensor:output:0;conv2d_flipout_1/Normal/sample/strided_slice/stack:output:0=conv2d_flipout_1/Normal/sample/strided_slice/stack_1:output:0=conv2d_flipout_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
0conv2d_flipout_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*%
valueB"          @   h
&conv2d_flipout_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : ~
4conv2d_flipout_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
6conv2d_flipout_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
6conv2d_flipout_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
.conv2d_flipout_1/Normal/sample/strided_slice_1StridedSlice9conv2d_flipout_1/Normal/sample/shape_as_tensor_1:output:0=conv2d_flipout_1/Normal/sample/strided_slice_1/stack:output:0?conv2d_flipout_1/Normal/sample/strided_slice_1/stack_1:output:0?conv2d_flipout_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskr
/conv2d_flipout_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB t
1conv2d_flipout_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
,conv2d_flipout_1/Normal/sample/BroadcastArgsBroadcastArgs:conv2d_flipout_1/Normal/sample/BroadcastArgs/s0_1:output:05conv2d_flipout_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
.conv2d_flipout_1/Normal/sample/BroadcastArgs_1BroadcastArgs1conv2d_flipout_1/Normal/sample/BroadcastArgs:r0:07conv2d_flipout_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:x
.conv2d_flipout_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:l
*conv2d_flipout_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
%conv2d_flipout_1/Normal/sample/concatConcatV27conv2d_flipout_1/Normal/sample/concat/values_0:output:03conv2d_flipout_1/Normal/sample/BroadcastArgs_1:r0:03conv2d_flipout_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:}
8conv2d_flipout_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    
:conv2d_flipout_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Hconv2d_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal.conv2d_flipout_1/Normal/sample/concat:output:0*
T0**
_output_shapes
: @*
dtype0?
7conv2d_flipout_1/Normal/sample/normal/random_normal/mulMulQconv2d_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Cconv2d_flipout_1/Normal/sample/normal/random_normal/stddev:output:0*
T0**
_output_shapes
: @?
3conv2d_flipout_1/Normal/sample/normal/random_normalAddV2;conv2d_flipout_1/Normal/sample/normal/random_normal/mul:z:0Aconv2d_flipout_1/Normal/sample/normal/random_normal/mean:output:0*
T0**
_output_shapes
: @?
"conv2d_flipout_1/Normal/sample/mulMul7conv2d_flipout_1/Normal/sample/normal/random_normal:z:0&conv2d_flipout_1/Normal/sample/add:z:0*
T0**
_output_shapes
: @?
$conv2d_flipout_1/Normal/sample/add_1AddV2&conv2d_flipout_1/Normal/sample/mul:z:0$conv2d_flipout_1/zeros_like:output:0*
T0**
_output_shapes
: @?
,conv2d_flipout_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"          @   ?
&conv2d_flipout_1/Normal/sample/ReshapeReshape(conv2d_flipout_1/Normal/sample/add_1:z:05conv2d_flipout_1/Normal/sample/Reshape/shape:output:0*
T0*&
_output_shapes
: @?
&conv2d_flipout_1/Conv2D/ReadVariableOpReadVariableOp/conv2d_flipout_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
conv2d_flipout_1/Conv2DConv2Dactivation/Relu:activations:0.conv2d_flipout_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
c
conv2d_flipout_1/ShapeShapeactivation/Relu:activations:0*
T0*
_output_shapes
:n
$conv2d_flipout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_flipout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_flipout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
conv2d_flipout_1/strided_sliceStridedSliceconv2d_flipout_1/Shape:output:0-conv2d_flipout_1/strided_slice/stack:output:0/conv2d_flipout_1/strided_slice/stack_1:output:0/conv2d_flipout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maska
conv2d_flipout_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/ExpandDims
ExpandDims'conv2d_flipout_1/strided_slice:output:0(conv2d_flipout_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:y
&conv2d_flipout_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????r
(conv2d_flipout_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_flipout_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
 conv2d_flipout_1/strided_slice_1StridedSliceconv2d_flipout_1/Shape:output:0/conv2d_flipout_1/strided_slice_1/stack:output:01conv2d_flipout_1/strided_slice_1/stack_1:output:01conv2d_flipout_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskc
!conv2d_flipout_1/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/ExpandDims_1
ExpandDims)conv2d_flipout_1/strided_slice_1:output:0*conv2d_flipout_1/ExpandDims_1/dim:output:0*
T0*
_output_shapes
:^
conv2d_flipout_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/concatConcatV2$conv2d_flipout_1/ExpandDims:output:0&conv2d_flipout_1/ExpandDims_1:output:0%conv2d_flipout_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
<conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
:conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
:conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
6conv2d_flipout_1/rademacher/uniform/sanitize_seed/seedRandomUniformIntEconv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/shape:output:0Cconv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/min:output:0Cconv2d_flipout_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
@conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
@conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Yconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter?conv2d_flipout_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
@conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
<conv2d_flipout_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2 conv2d_flipout_1/concat:output:0_conv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0cconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Iconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/alg:output:0Iconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/min:output:0Iconv2d_flipout_1/rademacher/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:????????? *
dtype0	c
!conv2d_flipout_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout_1/rademacher/mulMul*conv2d_flipout_1/rademacher/mul/x:output:0Econv2d_flipout_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:????????? c
!conv2d_flipout_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
conv2d_flipout_1/rademacher/subSub#conv2d_flipout_1/rademacher/mul:z:0*conv2d_flipout_1/rademacher/sub/y:output:0*
T0	*'
_output_shapes
:????????? ?
 conv2d_flipout_1/rademacher/CastCast#conv2d_flipout_1/rademacher/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:????????? e
#conv2d_flipout_1/ExpandDims_2/inputConst*
_output_shapes
: *
dtype0*
value	B :@c
!conv2d_flipout_1/ExpandDims_2/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/ExpandDims_2
ExpandDims,conv2d_flipout_1/ExpandDims_2/input:output:0*conv2d_flipout_1/ExpandDims_2/dim:output:0*
T0*
_output_shapes
:`
conv2d_flipout_1/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
conv2d_flipout_1/concat_1ConcatV2$conv2d_flipout_1/ExpandDims:output:0&conv2d_flipout_1/ExpandDims_2:output:0'conv2d_flipout_1/concat_1/axis:output:0*
N*
T0*
_output_shapes
:?
>conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
<conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
<conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
8conv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntGconv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Econv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0Econv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Bconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Bconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
[conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounterAconv2d_flipout_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Bconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
>conv2d_flipout_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2"conv2d_flipout_1/concat_1:output:0aconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0econv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Kconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Kconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Kconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????@*
dtype0	e
#conv2d_flipout_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
!conv2d_flipout_1/rademacher_1/mulMul,conv2d_flipout_1/rademacher_1/mul/x:output:0Gconv2d_flipout_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????@e
#conv2d_flipout_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
!conv2d_flipout_1/rademacher_1/subSub%conv2d_flipout_1/rademacher_1/mul:z:0,conv2d_flipout_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????@?
"conv2d_flipout_1/rademacher_1/CastCast%conv2d_flipout_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????@c
!conv2d_flipout_1/ExpandDims_3/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_3
ExpandDims$conv2d_flipout_1/rademacher/Cast:y:0*conv2d_flipout_1/ExpandDims_3/dim:output:0*
T0*+
_output_shapes
:????????? c
!conv2d_flipout_1/ExpandDims_4/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_4
ExpandDims&conv2d_flipout_1/rademacher_1/Cast:y:0*conv2d_flipout_1/ExpandDims_4/dim:output:0*
T0*+
_output_shapes
:?????????@c
!conv2d_flipout_1/ExpandDims_5/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_5
ExpandDims&conv2d_flipout_1/ExpandDims_3:output:0*conv2d_flipout_1/ExpandDims_5/dim:output:0*
T0*/
_output_shapes
:????????? c
!conv2d_flipout_1/ExpandDims_6/dimConst*
_output_shapes
: *
dtype0*
value	B :?
conv2d_flipout_1/ExpandDims_6
ExpandDims&conv2d_flipout_1/ExpandDims_4:output:0*conv2d_flipout_1/ExpandDims_6/dim:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_flipout_1/mulMulactivation/Relu:activations:0&conv2d_flipout_1/ExpandDims_5:output:0*
T0*/
_output_shapes
:????????? ?
conv2d_flipout_1/Conv2D_1Conv2Dconv2d_flipout_1/mul:z:0/conv2d_flipout_1/Normal/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
?
conv2d_flipout_1/mul_1Mul"conv2d_flipout_1/Conv2D_1:output:0&conv2d_flipout_1/ExpandDims_6:output:0*
T0*/
_output_shapes
:?????????@?
conv2d_flipout_1/addAddV2 conv2d_flipout_1/Conv2D:output:0conv2d_flipout_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@?
=conv2d_flipout_1/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Rconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
cconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Tconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOp]conv2d_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:@*
dtype0?
Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:@?
Kconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Yconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
[conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
[conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Sconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0bconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0dconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0dconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Vconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Xconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Sconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgsaconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0\conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Qconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Lconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Xconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Zconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Qconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastTo\conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Uconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:@?
Sconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   @   ?
Mconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeZconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0\conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:@?
>conv2d_flipout_1/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:@?
8conv2d_flipout_1/IndependentDeterministic/sample/ReshapeReshapeVconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Gconv2d_flipout_1/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:@?
conv2d_flipout_1/BiasAddBiasAddconv2d_flipout_1/add:z:0Aconv2d_flipout_1/IndependentDeterministic/sample/Reshape:output:0*
T0*/
_output_shapes
:?????????@?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp?conv2d_flipout_1_normal_sample_softplus_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558743*
T0*
_output_shapes
: ?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp/conv2d_flipout_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558743*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?conv2d_flipout_1_kullbackleibler_independentnormal_constructed_at_conv2d_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558743*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*&
_output_shapes
: @?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*%
valueB"?????????????????
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
"conv2d_flipout_1/divergence_kernelIdentity?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: ?
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3!conv2d_flipout_1/BiasAdd:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( 
activation_1/ReluRelu*batch_normalization_1/FusedBatchNormV3:y:0*
T0*/
_output_shapes
:?????????@^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????@  ?
flatten/ReshapeReshapeactivation_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????y
(dense_flipout/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     c
dense_flipout/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_flipout/zeros_likeFill1dense_flipout/zeros_like/shape_as_tensor:output:0'dense_flipout/zeros_like/Const:output:0*
T0* 
_output_shapes
:
??k
(dense_flipout/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
3dense_flipout/Normal/sample/Softplus/ReadVariableOpReadVariableOp<dense_flipout_normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
$dense_flipout/Normal/sample/SoftplusSoftplus;dense_flipout/Normal/sample/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??f
!dense_flipout/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
dense_flipout/Normal/sample/addAddV2*dense_flipout/Normal/sample/add/x:output:02dense_flipout/Normal/sample/Softplus:activations:0*
T0* 
_output_shapes
:
??|
+dense_flipout/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     c
!dense_flipout/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : y
/dense_flipout/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1dense_flipout/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1dense_flipout/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
)dense_flipout/Normal/sample/strided_sliceStridedSlice4dense_flipout/Normal/sample/shape_as_tensor:output:08dense_flipout/Normal/sample/strided_slice/stack:output:0:dense_flipout/Normal/sample/strided_slice/stack_1:output:0:dense_flipout/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask~
-dense_flipout/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"@     e
#dense_flipout/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : {
1dense_flipout/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_flipout/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_flipout/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_flipout/Normal/sample/strided_slice_1StridedSlice6dense_flipout/Normal/sample/shape_as_tensor_1:output:0:dense_flipout/Normal/sample/strided_slice_1/stack:output:0<dense_flipout/Normal/sample/strided_slice_1/stack_1:output:0<dense_flipout/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masko
,dense_flipout/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB q
.dense_flipout/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
)dense_flipout/Normal/sample/BroadcastArgsBroadcastArgs7dense_flipout/Normal/sample/BroadcastArgs/s0_1:output:02dense_flipout/Normal/sample/strided_slice:output:0*
_output_shapes
:?
+dense_flipout/Normal/sample/BroadcastArgs_1BroadcastArgs.dense_flipout/Normal/sample/BroadcastArgs:r0:04dense_flipout/Normal/sample/strided_slice_1:output:0*
_output_shapes
:u
+dense_flipout/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:i
'dense_flipout/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
"dense_flipout/Normal/sample/concatConcatV24dense_flipout/Normal/sample/concat/values_0:output:00dense_flipout/Normal/sample/BroadcastArgs_1:r0:00dense_flipout/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:z
5dense_flipout/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    |
7dense_flipout/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Edense_flipout/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal+dense_flipout/Normal/sample/concat:output:0*
T0*$
_output_shapes
:??*
dtype0?
4dense_flipout/Normal/sample/normal/random_normal/mulMulNdense_flipout/Normal/sample/normal/random_normal/RandomStandardNormal:output:0@dense_flipout/Normal/sample/normal/random_normal/stddev:output:0*
T0*$
_output_shapes
:???
0dense_flipout/Normal/sample/normal/random_normalAddV28dense_flipout/Normal/sample/normal/random_normal/mul:z:0>dense_flipout/Normal/sample/normal/random_normal/mean:output:0*
T0*$
_output_shapes
:???
dense_flipout/Normal/sample/mulMul4dense_flipout/Normal/sample/normal/random_normal:z:0#dense_flipout/Normal/sample/add:z:0*
T0*$
_output_shapes
:???
!dense_flipout/Normal/sample/add_1AddV2#dense_flipout/Normal/sample/mul:z:0!dense_flipout/zeros_like:output:0*
T0*$
_output_shapes
:??z
)dense_flipout/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     ?
#dense_flipout/Normal/sample/ReshapeReshape%dense_flipout/Normal/sample/add_1:z:02dense_flipout/Normal/sample/Reshape/shape:output:0*
T0* 
_output_shapes
:
??[
dense_flipout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:k
!dense_flipout/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: v
#dense_flipout/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????m
#dense_flipout/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_flipout/strided_sliceStridedSlicedense_flipout/Shape:output:0*dense_flipout/strided_slice/stack:output:0,dense_flipout/strided_slice/stack_1:output:0,dense_flipout/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
9dense_flipout/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
7dense_flipout/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????}
7dense_flipout/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
3dense_flipout/rademacher/uniform/sanitize_seed/seedRandomUniformIntBdense_flipout/rademacher/uniform/sanitize_seed/seed/shape:output:0@dense_flipout/rademacher/uniform/sanitize_seed/seed/min:output:0@dense_flipout/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:
=dense_flipout/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R 
=dense_flipout/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Vdense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter<dense_flipout/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::
=dense_flipout/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
9dense_flipout/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout/Shape:output:0\dense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0`dense_flipout/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Fdense_flipout/rademacher/uniform/stateless_random_uniform/alg:output:0Fdense_flipout/rademacher/uniform/stateless_random_uniform/min:output:0Fdense_flipout/rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	`
dense_flipout/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher/mulMul'dense_flipout/rademacher/mul/x:output:0Bdense_flipout/rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????`
dense_flipout/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher/subSub dense_flipout/rademacher/mul:z:0'dense_flipout/rademacher/sub/y:output:0*
T0	*(
_output_shapes
:???????????
dense_flipout/rademacher/CastCast dense_flipout/rademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????a
dense_flipout/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value
B :?^
dense_flipout/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout/ExpandDims
ExpandDims'dense_flipout/ExpandDims/input:output:0%dense_flipout/ExpandDims/dim:output:0*
T0*
_output_shapes
:[
dense_flipout/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout/concatConcatV2$dense_flipout/strided_slice:output:0!dense_flipout/ExpandDims:output:0"dense_flipout/concat/axis:output:0*
N*
T0*
_output_shapes
:?
;dense_flipout/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9dense_flipout/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????
9dense_flipout/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
5dense_flipout/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntDdense_flipout/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Bdense_flipout/rademacher_1/uniform/sanitize_seed/seed/min:output:0Bdense_flipout/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
?dense_flipout/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
?dense_flipout/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Xdense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter>dense_flipout/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
?dense_flipout/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
;dense_flipout/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout/concat:output:0^dense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0bdense_flipout/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hdense_flipout/rademacher_1/uniform/stateless_random_uniform/alg:output:0Hdense_flipout/rademacher_1/uniform/stateless_random_uniform/min:output:0Hdense_flipout/rademacher_1/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	b
 dense_flipout/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher_1/mulMul)dense_flipout/rademacher_1/mul/x:output:0Ddense_flipout/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????b
 dense_flipout/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout/rademacher_1/subSub"dense_flipout/rademacher_1/mul:z:0)dense_flipout/rademacher_1/sub/y:output:0*
T0	*(
_output_shapes
:???????????
dense_flipout/rademacher_1/CastCast"dense_flipout/rademacher_1/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:???????????
dense_flipout/mulMulflatten/Reshape:output:0!dense_flipout/rademacher/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_flipout/MatMulMatMuldense_flipout/mul:z:0,dense_flipout/Normal/sample/Reshape:output:0*
T0*(
_output_shapes
:???????????
dense_flipout/mul_1Muldense_flipout/MatMul:product:0#dense_flipout/rademacher_1/Cast:y:0*
T0*(
_output_shapes
:???????????
%dense_flipout/MatMul_1/ReadVariableOpReadVariableOp.dense_flipout_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_flipout/MatMul_1MatMulflatten/Reshape:output:0-dense_flipout/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_flipout/addAddV2 dense_flipout/MatMul_1:product:0dense_flipout/mul_1:z:0*
T0*(
_output_shapes
:??????????}
:dense_flipout/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Odense_flipout/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
`dense_flipout/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Qdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOpZdense_flipout_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:??
Hdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Vdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Xdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Xdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Pdense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice[dense_flipout/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0_dense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0adense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0adense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Sdense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Udense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Pdense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs^dense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0Ydense_flipout/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Ndense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Idense_flipout/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2[dense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Udense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0[dense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Wdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Ndense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastToYdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Rdense_flipout/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes
:	??
Pdense_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
Jdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeWdense_flipout/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0Ydense_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	??
;dense_flipout/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
5dense_flipout/IndependentDeterministic/sample/ReshapeReshapeSdense_flipout/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Ddense_flipout/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes	
:??
dense_flipout/BiasAddBiasAdddense_flipout/add:z:0>dense_flipout/IndependentDeterministic/sample/Reshape:output:0*
T0*(
_output_shapes
:??????????m
dense_flipout/ReluReludense_flipout/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp<dense_flipout_normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558895*
T0*
_output_shapes
: ?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp.dense_flipout_matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558895*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?dense_flipout_kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2558895*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
dense_flipout/divergence_kernelIdentity?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: {
*dense_flipout_1/zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   e
 dense_flipout_1/zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ?
dense_flipout_1/zeros_likeFill3dense_flipout_1/zeros_like/shape_as_tensor:output:0)dense_flipout_1/zeros_like/Const:output:0*
T0*
_output_shapes
:	?
m
*dense_flipout_1/Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
5dense_flipout_1/Normal/sample/Softplus/ReadVariableOpReadVariableOp>dense_flipout_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
&dense_flipout_1/Normal/sample/SoftplusSoftplus=dense_flipout_1/Normal/sample/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
h
#dense_flipout_1/Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
!dense_flipout_1/Normal/sample/addAddV2,dense_flipout_1/Normal/sample/add/x:output:04dense_flipout_1/Normal/sample/Softplus:activations:0*
T0*
_output_shapes
:	?
~
-dense_flipout_1/Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"   
   e
#dense_flipout_1/Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : {
1dense_flipout_1/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3dense_flipout_1/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3dense_flipout_1/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
+dense_flipout_1/Normal/sample/strided_sliceStridedSlice6dense_flipout_1/Normal/sample/shape_as_tensor:output:0:dense_flipout_1/Normal/sample/strided_slice/stack:output:0<dense_flipout_1/Normal/sample/strided_slice/stack_1:output:0<dense_flipout_1/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
/dense_flipout_1/Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"   
   g
%dense_flipout_1/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : }
3dense_flipout_1/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5dense_flipout_1/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5dense_flipout_1/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
-dense_flipout_1/Normal/sample/strided_slice_1StridedSlice8dense_flipout_1/Normal/sample/shape_as_tensor_1:output:0<dense_flipout_1/Normal/sample/strided_slice_1/stack:output:0>dense_flipout_1/Normal/sample/strided_slice_1/stack_1:output:0>dense_flipout_1/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskq
.dense_flipout_1/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB s
0dense_flipout_1/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
+dense_flipout_1/Normal/sample/BroadcastArgsBroadcastArgs9dense_flipout_1/Normal/sample/BroadcastArgs/s0_1:output:04dense_flipout_1/Normal/sample/strided_slice:output:0*
_output_shapes
:?
-dense_flipout_1/Normal/sample/BroadcastArgs_1BroadcastArgs0dense_flipout_1/Normal/sample/BroadcastArgs:r0:06dense_flipout_1/Normal/sample/strided_slice_1:output:0*
_output_shapes
:w
-dense_flipout_1/Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:k
)dense_flipout_1/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
$dense_flipout_1/Normal/sample/concatConcatV26dense_flipout_1/Normal/sample/concat/values_0:output:02dense_flipout_1/Normal/sample/BroadcastArgs_1:r0:02dense_flipout_1/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:|
7dense_flipout_1/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    ~
9dense_flipout_1/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
Gdense_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal-dense_flipout_1/Normal/sample/concat:output:0*
T0*#
_output_shapes
:?
*
dtype0?
6dense_flipout_1/Normal/sample/normal/random_normal/mulMulPdense_flipout_1/Normal/sample/normal/random_normal/RandomStandardNormal:output:0Bdense_flipout_1/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:?
?
2dense_flipout_1/Normal/sample/normal/random_normalAddV2:dense_flipout_1/Normal/sample/normal/random_normal/mul:z:0@dense_flipout_1/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:?
?
!dense_flipout_1/Normal/sample/mulMul6dense_flipout_1/Normal/sample/normal/random_normal:z:0%dense_flipout_1/Normal/sample/add:z:0*
T0*#
_output_shapes
:?
?
#dense_flipout_1/Normal/sample/add_1AddV2%dense_flipout_1/Normal/sample/mul:z:0#dense_flipout_1/zeros_like:output:0*
T0*#
_output_shapes
:?
|
+dense_flipout_1/Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
%dense_flipout_1/Normal/sample/ReshapeReshape'dense_flipout_1/Normal/sample/add_1:z:04dense_flipout_1/Normal/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	?
e
dense_flipout_1/ShapeShape dense_flipout/Relu:activations:0*
T0*
_output_shapes
:m
#dense_flipout_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: x
%dense_flipout_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????o
%dense_flipout_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
dense_flipout_1/strided_sliceStridedSlicedense_flipout_1/Shape:output:0,dense_flipout_1/strided_slice/stack:output:0.dense_flipout_1/strided_slice/stack_1:output:0.dense_flipout_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
;dense_flipout_1/rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
9dense_flipout_1/rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????
9dense_flipout_1/rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
5dense_flipout_1/rademacher/uniform/sanitize_seed/seedRandomUniformIntDdense_flipout_1/rademacher/uniform/sanitize_seed/seed/shape:output:0Bdense_flipout_1/rademacher/uniform/sanitize_seed/seed/min:output:0Bdense_flipout_1/rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
?dense_flipout_1/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
?dense_flipout_1/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Xdense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter>dense_flipout_1/rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
?dense_flipout_1/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
;dense_flipout_1/rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout_1/Shape:output:0^dense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0bdense_flipout_1/rademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Hdense_flipout_1/rademacher/uniform/stateless_random_uniform/alg:output:0Hdense_flipout_1/rademacher/uniform/stateless_random_uniform/min:output:0Hdense_flipout_1/rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	b
 dense_flipout_1/rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout_1/rademacher/mulMul)dense_flipout_1/rademacher/mul/x:output:0Ddense_flipout_1/rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????b
 dense_flipout_1/rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
dense_flipout_1/rademacher/subSub"dense_flipout_1/rademacher/mul:z:0)dense_flipout_1/rademacher/sub/y:output:0*
T0	*(
_output_shapes
:???????????
dense_flipout_1/rademacher/CastCast"dense_flipout_1/rademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????b
 dense_flipout_1/ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value	B :
`
dense_flipout_1/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout_1/ExpandDims
ExpandDims)dense_flipout_1/ExpandDims/input:output:0'dense_flipout_1/ExpandDims/dim:output:0*
T0*
_output_shapes
:]
dense_flipout_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
dense_flipout_1/concatConcatV2&dense_flipout_1/strided_slice:output:0#dense_flipout_1/ExpandDims:output:0$dense_flipout_1/concat/axis:output:0*
N*
T0*
_output_shapes
:?
=dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
;dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
??????????
;dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
7dense_flipout_1/rademacher_1/uniform/sanitize_seed/seedRandomUniformIntFdense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/shape:output:0Ddense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/min:output:0Ddense_flipout_1/rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:?
Adense_flipout_1/rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R ?
Adense_flipout_1/rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Zdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter@dense_flipout_1/rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::?
Adense_flipout_1/rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
=dense_flipout_1/rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2dense_flipout_1/concat:output:0`dense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0ddense_flipout_1/rademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0Jdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/alg:output:0Jdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/min:output:0Jdense_flipout_1/rademacher_1/uniform/stateless_random_uniform/max:output:0*'
_output_shapes
:?????????
*
dtype0	d
"dense_flipout_1/rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
 dense_flipout_1/rademacher_1/mulMul+dense_flipout_1/rademacher_1/mul/x:output:0Fdense_flipout_1/rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*'
_output_shapes
:?????????
d
"dense_flipout_1/rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R?
 dense_flipout_1/rademacher_1/subSub$dense_flipout_1/rademacher_1/mul:z:0+dense_flipout_1/rademacher_1/sub/y:output:0*
T0	*'
_output_shapes
:?????????
?
!dense_flipout_1/rademacher_1/CastCast$dense_flipout_1/rademacher_1/sub:z:0*

DstT0*

SrcT0	*'
_output_shapes
:?????????
?
dense_flipout_1/mulMul dense_flipout/Relu:activations:0#dense_flipout_1/rademacher/Cast:y:0*
T0*(
_output_shapes
:???????????
dense_flipout_1/MatMulMatMuldense_flipout_1/mul:z:0.dense_flipout_1/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
dense_flipout_1/mul_1Mul dense_flipout_1/MatMul:product:0%dense_flipout_1/rademacher_1/Cast:y:0*
T0*'
_output_shapes
:?????????
?
'dense_flipout_1/MatMul_1/ReadVariableOpReadVariableOp0dense_flipout_1_matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
dense_flipout_1/MatMul_1MatMul dense_flipout/Relu:activations:0/dense_flipout_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
?
dense_flipout_1/addAddV2"dense_flipout_1/MatMul_1:product:0dense_flipout_1/mul_1:z:0*
T0*'
_output_shapes
:?????????

<dense_flipout_1/IndependentDeterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
Qdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
bdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
Sdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpReadVariableOp\dense_flipout_1_independentdeterministic_sample_deterministic_sample_readvariableop_resource*
_output_shapes
:
*
dtype0?
Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
?
Jdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
Xdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
Zdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
Zdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Rdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_sliceStridedSlice]dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/shape_as_tensor:output:0adense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack:output:0cdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_1:output:0cdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
Udense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
Wdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Rdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgsBroadcastArgs`dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0[dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
Pdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Kdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concatConcatV2]dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_0:output:0Wdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastArgs:r0:0]dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/values_2:output:0Ydense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
Pdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastToBroadcastTo[dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp:value:0Tdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes

:
?
Rdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"   
   ?
Ldense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReshapeReshapeYdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/BroadcastTo:output:0[dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes

:
?
=dense_flipout_1/IndependentDeterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
7dense_flipout_1/IndependentDeterministic/sample/ReshapeReshapeUdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/Reshape:output:0Fdense_flipout_1/IndependentDeterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:
?
dense_flipout_1/BiasAddBiasAdddense_flipout_1/add:z:0@dense_flipout_1/IndependentDeterministic/sample/Reshape:output:0*
T0*'
_output_shapes
:?????????
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp>dense_flipout_1_normal_sample_softplus_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559029*
T0*
_output_shapes
: ?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp0dense_flipout_1_matmul_1_readvariableop_resource*
_output_shapes
:	?
*
dtype0?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559029*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?dense_flipout_1_kullbackleibler_independentnormal_constructed_at_dense_flipout_1_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2559029*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0*
_output_shapes
:	?
?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
!dense_flipout_1/divergence_kernelIdentity?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: o
IdentityIdentity dense_flipout_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????
i

Identity_1Identity)conv2d_flipout/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: k

Identity_2Identity+conv2d_flipout_1/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: h

Identity_3Identity(dense_flipout/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: j

Identity_4Identity*dense_flipout_1/divergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1%^conv2d_flipout/Conv2D/ReadVariableOpS^conv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp5^conv2d_flipout/Normal/sample/Softplus/ReadVariableOp'^conv2d_flipout_1/Conv2D/ReadVariableOpU^conv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp7^conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOpR^dense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp&^dense_flipout/MatMul_1/ReadVariableOp4^dense_flipout/Normal/sample/Softplus/ReadVariableOpT^dense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp?^dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp(^dense_flipout_1/MatMul_1/ReadVariableOp6^dense_flipout_1/Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12L
$conv2d_flipout/Conv2D/ReadVariableOp$conv2d_flipout/Conv2D/ReadVariableOp2?
Rconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpRconv2d_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?conv2d_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2l
4conv2d_flipout/Normal/sample/Softplus/ReadVariableOp4conv2d_flipout/Normal/sample/Softplus/ReadVariableOp2P
&conv2d_flipout_1/Conv2D/ReadVariableOp&conv2d_flipout_1/Conv2D/ReadVariableOp2?
Tconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpTconv2d_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?conv2d_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_conv2d_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2p
6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp6conv2d_flipout_1/Normal/sample/Softplus/ReadVariableOp2?
Qdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpQdense_flipout/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?dense_flipout/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2N
%dense_flipout/MatMul_1/ReadVariableOp%dense_flipout/MatMul_1/ReadVariableOp2j
3dense_flipout/Normal/sample/Softplus/ReadVariableOp3dense_flipout/Normal/sample/Softplus/ReadVariableOp2?
Sdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOpSdense_flipout_1/IndependentDeterministic/sample/Deterministic/sample/ReadVariableOp2?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?dense_flipout_1/KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout_1/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp2R
'dense_flipout_1/MatMul_1/ReadVariableOp'dense_flipout_1/MatMul_1/ReadVariableOp2n
5dense_flipout_1/Normal/sample/Softplus/ReadVariableOp5dense_flipout_1/Normal/sample/Softplus/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?
?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557007

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?=
?
G__inference_sequential_layer_call_and_return_conditional_losses_2557744

inputs0
conv2d_flipout_2557212: 0
conv2d_flipout_2557214: $
conv2d_flipout_2557216: 
conv2d_flipout_2557218
conv2d_flipout_2557220)
batch_normalization_2557224: )
batch_normalization_2557226: )
batch_normalization_2557228: )
batch_normalization_2557230: 2
conv2d_flipout_1_2557398: @2
conv2d_flipout_1_2557400: @&
conv2d_flipout_1_2557402:@
conv2d_flipout_1_2557404
conv2d_flipout_1_2557406+
batch_normalization_1_2557410:@+
batch_normalization_1_2557412:@+
batch_normalization_1_2557414:@+
batch_normalization_1_2557416:@)
dense_flipout_2557575:
??)
dense_flipout_2557577:
??$
dense_flipout_2557579:	?
dense_flipout_2557581
dense_flipout_2557583*
dense_flipout_1_2557727:	?
*
dense_flipout_1_2557729:	?
%
dense_flipout_1_2557731:

dense_flipout_1_2557733
dense_flipout_1_2557735
identity

identity_1

identity_2

identity_3

identity_4??+batch_normalization/StatefulPartitionedCall?-batch_normalization_1/StatefulPartitionedCall?&conv2d_flipout/StatefulPartitionedCall?(conv2d_flipout_1/StatefulPartitionedCall?%dense_flipout/StatefulPartitionedCall?'dense_flipout_1/StatefulPartitionedCall?
&conv2d_flipout/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_flipout_2557212conv2d_flipout_2557214conv2d_flipout_2557216conv2d_flipout_2557218conv2d_flipout_2557220*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:????????? : *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2557211?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall/conv2d_flipout/StatefulPartitionedCall:output:0batch_normalization_2557224batch_normalization_2557226batch_normalization_2557228batch_normalization_2557230*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556943?
activation/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2557238?
(conv2d_flipout_1/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0conv2d_flipout_1_2557398conv2d_flipout_1_2557400conv2d_flipout_1_2557402conv2d_flipout_1_2557404conv2d_flipout_1_2557406*
Tin

2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????@: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *V
fQRO
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2557397?
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_flipout_1/StatefulPartitionedCall:output:0batch_normalization_1_2557410batch_normalization_1_2557412batch_normalization_1_2557414batch_normalization_1_2557416*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2557007?
activation_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2557424?
flatten/PartitionedCallPartitionedCall%activation_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2557432?
%dense_flipout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_flipout_2557575dense_flipout_2557577dense_flipout_2557579dense_flipout_2557581dense_flipout_2557583*
Tin

2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:??????????: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2557574?
'dense_flipout_1/StatefulPartitionedCallStatefulPartitionedCall.dense_flipout/StatefulPartitionedCall:output:0dense_flipout_1_2557727dense_flipout_1_2557729dense_flipout_1_2557731dense_flipout_1_2557733dense_flipout_1_2557735*
Tin

2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:?????????
: *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2557726
IdentityIdentity0dense_flipout_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
o

Identity_1Identity/conv2d_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: q

Identity_2Identity1conv2d_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: n

Identity_3Identity.dense_flipout/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: p

Identity_4Identity0dense_flipout_1/StatefulPartitionedCall:output:1^NoOp*
T0*
_output_shapes
: ?
NoOpNoOp,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall'^conv2d_flipout/StatefulPartitionedCall)^conv2d_flipout_1/StatefulPartitionedCall&^dense_flipout/StatefulPartitionedCall(^dense_flipout_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*(
_construction_contextkEagerRuntime*?
_input_shapes?
?:?????????: : : : : : : : : : : : : : @: : : : : : : : :
??: : : : :	?
2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2P
&conv2d_flipout/StatefulPartitionedCall&conv2d_flipout/StatefulPartitionedCall2T
(conv2d_flipout_1/StatefulPartitionedCall(conv2d_flipout_1/StatefulPartitionedCall2N
%dense_flipout/StatefulPartitionedCall%dense_flipout/StatefulPartitionedCall2R
'dense_flipout_1/StatefulPartitionedCall'dense_flipout_1/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs:

_output_shapes
: :,(
&
_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: @:

_output_shapes
: :&"
 
_output_shapes
:
??:

_output_shapes
: :%!

_output_shapes
:	?

?	
?
5__inference_batch_normalization_layer_call_fn_2559931

inputs
unknown: 
	unknown_0: 
	unknown_1: 
	unknown_2: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2556974?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+??????????????????????????? `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+??????????????????????????? : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2560389

inputsB
.normal_sample_softplus_readvariableop_resource:
??4
 matmul_1_readvariableop_resource:
??x
iindependentdeterministic_constructed_at_dense_flipout_sample_deterministic_sample_readvariableop_resource:	??
?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560361?
?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x
identity

identity_1??`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp??KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?MatMul_1/ReadVariableOp?%Normal/sample/Softplus/ReadVariableOpk
zeros_like/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     U
zeros_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    }

zeros_likeFill#zeros_like/shape_as_tensor:output:0zeros_like/Const:output:0*
T0* 
_output_shapes
:
??]
Normal/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
%Normal/sample/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0|
Normal/sample/SoftplusSoftplus-Normal/sample/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??X
Normal/sample/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
Normal/sample/addAddV2Normal/sample/add/x:output:0$Normal/sample/Softplus:activations:0*
T0* 
_output_shapes
:
??n
Normal/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB"@     U
Normal/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskp
Normal/sample/shape_as_tensor_1Const*
_output_shapes
:*
dtype0*
valueB"@     W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
:?
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
:g
Normal/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ???
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*$
_output_shapes
:??*
dtype0?
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*$
_output_shapes
:???
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*$
_output_shapes
:???
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/sample/add:z:0*
T0*$
_output_shapes
:??w
Normal/sample/add_1AddV2Normal/sample/mul:z:0zeros_like:output:0*
T0*$
_output_shapes
:??l
Normal/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"@     ?
Normal/sample/ReshapeReshapeNormal/sample/add_1:z:0$Normal/sample/Reshape/shape:output:0*
T0* 
_output_shapes
:
??;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: h
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?????????_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masku
+rademacher/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:t
)rademacher/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????o
)rademacher/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
%rademacher/uniform/sanitize_seed/seedRandomUniformInt4rademacher/uniform/sanitize_seed/seed/shape:output:02rademacher/uniform/sanitize_seed/seed/min:output:02rademacher/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:q
/rademacher/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R q
/rademacher/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Hrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter.rademacher/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::q
/rademacher/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
+rademacher/uniform/stateless_random_uniformStatelessRandomUniformIntV2Shape:output:0Nrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Rrademacher/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:08rademacher/uniform/stateless_random_uniform/alg:output:08rademacher/uniform/stateless_random_uniform/min:output:08rademacher/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	R
rademacher/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher/mulMulrademacher/mul/x:output:04rademacher/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????R
rademacher/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rw
rademacher/subSubrademacher/mul:z:0rademacher/sub/y:output:0*
T0	*(
_output_shapes
:??????????m
rademacher/CastCastrademacher/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????S
ExpandDims/inputConst*
_output_shapes
: *
dtype0*
value
B :?P
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : q

ExpandDims
ExpandDimsExpandDims/input:output:0ExpandDims/dim:output:0*
T0*
_output_shapes
:M
concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
concatConcatV2strided_slice:output:0ExpandDims:output:0concat/axis:output:0*
N*
T0*
_output_shapes
:w
-rademacher_1/uniform/sanitize_seed/seed/shapeConst*
_output_shapes
:*
dtype0*
valueB:v
+rademacher_1/uniform/sanitize_seed/seed/minConst*
_output_shapes
: *
dtype0*
valueB :
?????????q
+rademacher_1/uniform/sanitize_seed/seed/maxConst*
_output_shapes
: *
dtype0*
valueB :?????
'rademacher_1/uniform/sanitize_seed/seedRandomUniformInt6rademacher_1/uniform/sanitize_seed/seed/shape:output:04rademacher_1/uniform/sanitize_seed/seed/min:output:04rademacher_1/uniform/sanitize_seed/seed/max:output:0*
T0*

Tout0*
_output_shapes
:s
1rademacher_1/uniform/stateless_random_uniform/minConst*
_output_shapes
: *
dtype0	*
value	B	 R s
1rademacher_1/uniform/stateless_random_uniform/maxConst*
_output_shapes
: *
dtype0	*
value	B	 R?
Jrademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounterStatelessRandomGetKeyCounter0rademacher_1/uniform/sanitize_seed/seed:output:0*
Tseed0* 
_output_shapes
::s
1rademacher_1/uniform/stateless_random_uniform/algConst*
_output_shapes
: *
dtype0*
value	B :?
-rademacher_1/uniform/stateless_random_uniformStatelessRandomUniformIntV2concat:output:0Prademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:key:0Trademacher_1/uniform/stateless_random_uniform/StatelessRandomGetKeyCounter:counter:0:rademacher_1/uniform/stateless_random_uniform/alg:output:0:rademacher_1/uniform/stateless_random_uniform/min:output:0:rademacher_1/uniform/stateless_random_uniform/max:output:0*(
_output_shapes
:??????????*
dtype0	T
rademacher_1/mul/xConst*
_output_shapes
: *
dtype0	*
value	B	 R?
rademacher_1/mulMulrademacher_1/mul/x:output:06rademacher_1/uniform/stateless_random_uniform:output:0*
T0	*(
_output_shapes
:??????????T
rademacher_1/sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R}
rademacher_1/subSubrademacher_1/mul:z:0rademacher_1/sub/y:output:0*
T0	*(
_output_shapes
:??????????q
rademacher_1/CastCastrademacher_1/sub:z:0*

DstT0*

SrcT0	*(
_output_shapes
:??????????Z
mulMulinputsrademacher/Cast:y:0*
T0*(
_output_shapes
:??????????l
MatMulMatMulmul:z:0Normal/sample/Reshape:output:0*
T0*(
_output_shapes
:??????????h
mul_1MulMatMul:product:0rademacher_1/Cast:y:0*
T0*(
_output_shapes
:??????????z
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0n
MatMul_1MatMulinputsMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????^
addAddV2MatMul_1:product:0	mul_1:z:0*
T0*(
_output_shapes
:???????????
IIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
valueB ?
^IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/sample_shapeConst*
_output_shapes
: *
dtype0*
value	B :?
oIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/expand_to_vector/sample_shapeConst*
_output_shapes
:*
dtype0*
valueB:?
`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOpReadVariableOpiindependentdeterministic_constructed_at_dense_flipout_sample_deterministic_sample_readvariableop_resource*
_output_shapes	
:?*
dtype0?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:??
WIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ConstConst*
_output_shapes
: *
dtype0*
value	B : ?
eIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ?
gIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:?
gIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_sliceStridedSlicejIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/shape_as_tensor:output:0nIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack:output:0pIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_1:output:0pIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask?
bIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB ?
dIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB ?
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgsBroadcastArgsmIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs/s0_1:output:0hIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/strided_slice:output:0*
_output_shapes
:?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:?
aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_2Const*
_output_shapes
: *
dtype0*
valueB ?
]IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ?
XIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concatConcatV2jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_0:output:0dIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastArgs:r0:0jIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/values_2:output:0fIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:?
]IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastToBroadcastTohIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp:value:0aIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/concat:output:0*
T0*
_output_shapes
:	??
_IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      ?
YIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReshapeReshapefIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/BroadcastTo:output:0hIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape/shape:output:0*
T0*
_output_shapes
:	??
JIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:??
DIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/ReshapeReshapebIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/Reshape:output:0SIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape/shape:output:0*
T0*
_output_shapes	
:??
BiasAddBiasAddadd:z:0MIndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Reshape:output:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:???????????
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOpReadVariableOp.normal_sample_softplus_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/SoftplusSoftplus?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp:value:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *   4?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus:activations:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/LogLog?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/add:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1Log?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560361*
T0*
_output_shapes
: ?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/subSub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log:y:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log_1:y:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truedivRealDiv?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp:value:0?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560361*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1RealDiv?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_truediv_1_x?kullbackleibler_independentnormal_constructed_at_dense_flipout_kullbackleibler_a_independentnormal_kullbackleibler_b_kullbackleibler_normal_kullbackleibler_a_normal_kullbackleibler_b_kullbackleibler_2560361*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifferenceSquaredDifference?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/truediv_1:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mulMul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/SquaredDifference:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1Expm1?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_1:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ??
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2Mul?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2/x:output:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Expm1:y:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/addAddV2?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/mul_2:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1Sub?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/add:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub:z:0*
T0* 
_output_shapes
:
???
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"?????????
xKullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/SumSum?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/sub_1:z:0?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum/reduction_indices:output:0*
T0*
_output_shapes
: ?
divergence_kernelIdentity?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/Sum:output:0*
T0*
_output_shapes
: b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????Z

Identity_1Identitydivergence_kernel:output:0^NoOp*
T0*
_output_shapes
: ?
NoOpNoOpa^IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?^KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp^MatMul_1/ReadVariableOp&^Normal/sample/Softplus/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????: : : : :
??2?
`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp`IndependentDeterministic_CONSTRUCTED_AT_dense_flipout/sample/Deterministic/sample/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/Log/Softplus/ReadVariableOp2?
?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp?KullbackLeibler/IndependentNormal_CONSTRUCTED_AT_dense_flipout/KullbackLeibler_a/IndependentNormal/KullbackLeibler_b/KullbackLeibler/Normal/KullbackLeibler_a/Normal/KullbackLeibler_b/KullbackLeibler/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2N
%Normal/sample/Softplus/ReadVariableOp%Normal/sample/Softplus/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:

_output_shapes
: :&"
 
_output_shapes
:
??"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
]
conv2d_flipout_inputE
&serving_default_conv2d_flipout_input:0?????????C
dense_flipout_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
?
kernel_posterior_loc
($kernel_posterior_untransformed_scale
kernel_posterior
kernel_prior
bias_posterior_loc
bias_posterior
kernel_posterior_affine
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
?
axis
	 gamma
!beta
"moving_mean
#moving_variance
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses"
_tf_keras_layer
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0kernel_posterior_loc
(1$kernel_posterior_untransformed_scale
2kernel_posterior
3kernel_prior
4bias_posterior_loc
5bias_posterior
6kernel_posterior_affine
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
?
=axis
	>gamma
?beta
@moving_mean
Amoving_variance
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses"
_tf_keras_layer
?
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
?
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Tkernel_posterior_loc
(U$kernel_posterior_untransformed_scale
Vkernel_posterior
Wkernel_prior
Xbias_posterior_loc
Ybias_posterior
Zkernel_posterior_affine
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
?
akernel_posterior_loc
(b$kernel_posterior_untransformed_scale
ckernel_posterior
dkernel_prior
ebias_posterior_loc
fbias_posterior
gkernel_posterior_affine
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0
1
2
 3
!4
"5
#6
07
18
49
>10
?11
@12
A13
T14
U15
X16
a17
b18
e19"
trackable_list_wrapper
?
0
1
2
 3
!4
05
16
47
>8
?9
T10
U11
X12
a13
b14
e15"
trackable_list_wrapper
 "
trackable_list_wrapper
?
nnon_trainable_variables

olayers
pmetrics
qlayer_regularization_losses
rlayer_metrics

	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_sequential_layer_call_fn_2557807
,__inference_sequential_layer_call_fn_2558384
,__inference_sequential_layer_call_fn_2558449
,__inference_sequential_layer_call_fn_2558165?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_sequential_layer_call_and_return_conditional_losses_2559060
G__inference_sequential_layer_call_and_return_conditional_losses_2559671
G__inference_sequential_layer_call_and_return_conditional_losses_2558242
G__inference_sequential_layer_call_and_return_conditional_losses_2558319?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_2556921conv2d_flipout_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
,
sserving_default"
signature_map
=:; 2#conv2d_flipout/kernel_posterior_loc
M:K 23conv2d_flipout/kernel_posterior_untransformed_scale
E
t_distribution
u_graph_parents"
_generic_user_object
E
v_distribution
w_graph_parents"
_generic_user_object
/:- 2!conv2d_flipout/bias_posterior_loc
E
x_distribution
y_graph_parents"
_generic_user_object
>

z_scale
{_graph_parents"
_generic_user_object
5
0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
|non_trainable_variables

}layers
~metrics
layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?2?
0__inference_conv2d_flipout_layer_call_fn_2559750?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2559905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
<
 0
!1
"2
#3"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
?2?
5__inference_batch_normalization_layer_call_fn_2559918
5__inference_batch_normalization_layer_call_fn_2559931?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2559949
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2559967?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?2?
,__inference_activation_layer_call_fn_2559972?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_activation_layer_call_and_return_conditional_losses_2559977?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?:= @2%conv2d_flipout_1/kernel_posterior_loc
O:M @25conv2d_flipout_1/kernel_posterior_untransformed_scale
G
?_distribution
?_graph_parents"
_generic_user_object
G
?_distribution
?_graph_parents"
_generic_user_object
1:/@2#conv2d_flipout_1/bias_posterior_loc
G
?_distribution
?_graph_parents"
_generic_user_object
@
?_scale
?_graph_parents"
_generic_user_object
5
00
11
42"
trackable_list_wrapper
5
00
11
42"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
?2?
2__inference_conv2d_flipout_1_layer_call_fn_2559993?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2560150?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
):'@2batch_normalization_1/gamma
(:&@2batch_normalization_1/beta
1:/@ (2!batch_normalization_1/moving_mean
5:3@ (2%batch_normalization_1/moving_variance
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
?2?
7__inference_batch_normalization_1_layer_call_fn_2560163
7__inference_batch_normalization_1_layer_call_fn_2560176?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2560194
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2560212?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
?2?
.__inference_activation_1_layer_call_fn_2560217?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_activation_1_layer_call_and_return_conditional_losses_2560222?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
?2?
)__inference_flatten_layer_call_fn_2560227?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_flatten_layer_call_and_return_conditional_losses_2560233?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
6:4
??2"dense_flipout/kernel_posterior_loc
F:D
??22dense_flipout/kernel_posterior_untransformed_scale
G
?_distribution
?_graph_parents"
_generic_user_object
G
?_distribution
?_graph_parents"
_generic_user_object
/:-?2 dense_flipout/bias_posterior_loc
G
?_distribution
?_graph_parents"
_generic_user_object
@
?_scale
?_graph_parents"
_generic_user_object
5
T0
U1
X2"
trackable_list_wrapper
5
T0
U1
X2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
?2?
/__inference_dense_flipout_layer_call_fn_2560249?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2560389?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
7:5	?
2$dense_flipout_1/kernel_posterior_loc
G:E	?
24dense_flipout_1/kernel_posterior_untransformed_scale
G
?_distribution
?_graph_parents"
_generic_user_object
G
?_distribution
?_graph_parents"
_generic_user_object
0:.
2"dense_flipout_1/bias_posterior_loc
G
?_distribution
?_graph_parents"
_generic_user_object
@
?_scale
?_graph_parents"
_generic_user_object
5
a0
b1
e2"
trackable_list_wrapper
5
a0
b1
e2"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
?2?
1__inference_dense_flipout_1_layer_call_fn_2560405?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2560544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
<
"0
#1
@2
A3"
trackable_list_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
%__inference_signature_wrapper_2559734conv2d_flipout_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
I
_loc

z_scale
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
=
_loc
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
0_loc
?_scale
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
=
4_loc
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
1_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
T_loc
?_scale
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
=
X_loc
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
U_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
J
a_loc
?_scale
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
3
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
=
e_loc
?_graph_parents"
_generic_user_object
 "
trackable_list_wrapper
9
b_pretransformed_input"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_7?
"__inference__wrapped_model_2556921?$?? !"#104??>?@AUTX??bae??E?B
;?8
6?3
conv2d_flipout_input?????????
? "A?>
<
dense_flipout_1)?&
dense_flipout_1?????????
?
I__inference_activation_1_layer_call_and_return_conditional_losses_2560222h7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
.__inference_activation_1_layer_call_fn_2560217[7?4
-?*
(?%
inputs?????????@
? " ??????????@?
G__inference_activation_layer_call_and_return_conditional_losses_2559977h7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0????????? 
? ?
,__inference_activation_layer_call_fn_2559972[7?4
-?*
(?%
inputs????????? 
? " ?????????? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2560194?>?@AM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_2560212?>?@AM?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
7__inference_batch_normalization_1_layer_call_fn_2560163?>?@AM?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
7__inference_batch_normalization_1_layer_call_fn_2560176?>?@AM?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2559949? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "??<
5?2
0+??????????????????????????? 
? ?
P__inference_batch_normalization_layer_call_and_return_conditional_losses_2559967? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "??<
5?2
0+??????????????????????????? 
? ?
5__inference_batch_normalization_layer_call_fn_2559918? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p 
? "2?/+??????????????????????????? ?
5__inference_batch_normalization_layer_call_fn_2559931? !"#M?J
C?@
:?7
inputs+??????????????????????????? 
p
? "2?/+??????????????????????????? ?
M__inference_conv2d_flipout_1_layer_call_and_return_conditional_losses_2560150104??7?4
-?*
(?%
inputs????????? 
? ";?8
#? 
0?????????@
?
?	
1/0 ?
2__inference_conv2d_flipout_1_layer_call_fn_2559993d104??7?4
-?*
(?%
inputs????????? 
? " ??????????@?
K__inference_conv2d_flipout_layer_call_and_return_conditional_losses_2559905??7?4
-?*
(?%
inputs?????????
? ";?8
#? 
0????????? 
?
?	
1/0 ?
0__inference_conv2d_flipout_layer_call_fn_2559750d??7?4
-?*
(?%
inputs?????????
? " ?????????? ?
L__inference_dense_flipout_1_layer_call_and_return_conditional_losses_2560544pbae??0?-
&?#
!?
inputs??????????
? "3?0
?
0?????????

?
?	
1/0 ?
1__inference_dense_flipout_1_layer_call_fn_2560405Ubae??0?-
&?#
!?
inputs??????????
? "??????????
?
J__inference_dense_flipout_layer_call_and_return_conditional_losses_2560389qUTX??0?-
&?#
!?
inputs??????????
? "4?1
?
0??????????
?
?	
1/0 ?
/__inference_dense_flipout_layer_call_fn_2560249VUTX??0?-
&?#
!?
inputs??????????
? "????????????
D__inference_flatten_layer_call_and_return_conditional_losses_2560233a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0??????????
? ?
)__inference_flatten_layer_call_fn_2560227T7?4
-?*
(?%
inputs?????????@
? "????????????
G__inference_sequential_layer_call_and_return_conditional_losses_2558242?$?? !"#104??>?@AUTX??bae??M?J
C?@
6?3
conv2d_flipout_input?????????
p 

 
? "]?Z
?
0?????????

;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
G__inference_sequential_layer_call_and_return_conditional_losses_2558319?$?? !"#104??>?@AUTX??bae??M?J
C?@
6?3
conv2d_flipout_input?????????
p

 
? "]?Z
?
0?????????

;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
G__inference_sequential_layer_call_and_return_conditional_losses_2559060?$?? !"#104??>?@AUTX??bae????<
5?2
(?%
inputs?????????
p 

 
? "]?Z
?
0?????????

;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
G__inference_sequential_layer_call_and_return_conditional_losses_2559671?$?? !"#104??>?@AUTX??bae????<
5?2
(?%
inputs?????????
p

 
? "]?Z
?
0?????????

;?8
?	
1/0 
?	
1/1 
?	
1/2 
?	
1/3 ?
,__inference_sequential_layer_call_fn_2557807?$?? !"#104??>?@AUTX??bae??M?J
C?@
6?3
conv2d_flipout_input?????????
p 

 
? "??????????
?
,__inference_sequential_layer_call_fn_2558165?$?? !"#104??>?@AUTX??bae??M?J
C?@
6?3
conv2d_flipout_input?????????
p

 
? "??????????
?
,__inference_sequential_layer_call_fn_2558384?$?? !"#104??>?@AUTX??bae????<
5?2
(?%
inputs?????????
p 

 
? "??????????
?
,__inference_sequential_layer_call_fn_2558449?$?? !"#104??>?@AUTX??bae????<
5?2
(?%
inputs?????????
p

 
? "??????????
?
%__inference_signature_wrapper_2559734?$?? !"#104??>?@AUTX??bae??]?Z
? 
S?P
N
conv2d_flipout_input6?3
conv2d_flipout_input?????????"A?>
<
dense_flipout_1)?&
dense_flipout_1?????????
