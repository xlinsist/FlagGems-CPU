# FlagGems CPU迁移

## 进展

- benchmarks:
  
  - [x] blas
  - [x] norm

- tests: 大多数测试都能跑通，现有问题在下面的`TODO`中说明

  - [x] attension_ops
  - [ ] binary_pointwise_ops
    - [ ] clamp, nan_to_num算子：精度不足
    - [ ] `where` related ops：`AssertionError: CPU only. There seems a mistake to dispatch to here.`
    - [x] `floor_divide`，`remainder`相关测试报错：`Floating point exception`。
      该报错与Triton的对0取余行为有关。
    - [ ] `remainder`相关测试精度不足
  - [x] blas_ops
  - [x] distribution_ops
  - [x] general_reduction_ops
  - [x] libentry
  - [ ] norm_ops
    - [ ] `batch_norm`：精度不足
  - [x] pointwise_dynamic
  - [ ] pointwise_dynamic_type_promotion
    - [ ] `where` related ops：`AssertionError: CPU only. There seems a mistake to dispatch to here.`
  - [ ] reduction_ops
    - [ ] `batch_norm`：精度不足
    - [ ] 待添加
  - [x] shape_utils
  - [ ] special_ops
    - [ ] 精度不足: `pad`, `kron`
  - [x] tensor_constructor
    该测试中的`test_accuracy_randn`小概率出现精度不够的问题
  - [x] tensor_wrapper
  - [ ] unary_pointwise_ops
    - [ ] angle: RuntimeError: failed to translate module to LLVM IR
    - [ ] gelu: AssertionError: Tensor-likes are not close!
    - [ ] silu: triton.compiler.errors.CompilationError

## TODO

- [ ] 浮点精度问题：现在的triton-cpu应该只支持float和double精度的计算，目前FlagGems中的精度相关代码，如测试函数，尚未被修改。
- [ ] 并行：即使设置`TRITON_CPU_MAX_THREADS=0`，`pytest test_xxx.py`也无法并行运行算子。
- [ ] `Philox API`：对于`distribution ops`，采用了很多`cuda philox`，目前只采用了最简单的处理。
  相关函数：multinomial, randperm, rand_like, rand, randn, randn_like
- [ ] `libtuner`：`mm`算子采用`libtuner`，但是会报错。替换成`triton.autotuner`才能正常运行。
- [ ] `benchmark`运行时间过长，调整测试规模。
- [ ] 移除分布在几十个文件中多余的`torch_device_fn`，该变量使用场景相当局限和单一，可以考虑将该全局变量优化掉。
- [ ] 目前有几个测试的测试时间过长，即使设置`--mode=quick`，也需要至少十几分钟，甚至半小时，不利于正确性验证，考虑缩减测试规模或优化CPU上的相关算子。
  - [x] test_attension_ops
  - [ ] test_tensor_constructor
  - [ ] test_reduction_ops: 算子`conv`运行时间过长
  - [ ] test_special_ops: 算子`sort`, `upsample`运行时间过长
  - [ ] 待添加
- [ ] 优化`multinomial`算子。注：该算子存在改动且确定存在需要对CPU端进行更大的、进一步的优化，因此已将其放在cpu后端的算子目录中。
- [ ] 重新实现`where`算子，FlagGems的`where`算子实现不能用CPU跑。
- [ ] 目前对于`div`算子文件中`__remainder`函数的处理方式不一定正确，后续需要确定其正确性。同时`floordiv`相关函数也需要关注。

## 大致修改

1. 在`src/runtime/common_utils.py`中，加入了arm，不过我的测试设备是`Intel 13400`，这个目前无关紧要:

```python
class vendors(Enum):
    NVIDIA = 0
    CAMBRICON = 1
    METAX = 2
    ILUVATAR = 3
    MTHREADS = 4
    KUNLUNXIN = 5
    HYGON = 6
    AMD = 7
    AIPU = 8
    ARM = 9
```

2. 在`runtime`中，定义`get_torch_device_ctx`，代替在算子中大量使用的上下文相关cuda api，如`torch_device_fn.device(device)`。
3. 少量测试和框架代码使用到了cuda专用api`_DeviceGuard()`和`Philox`相关的cuda api，已经做了简易兼容。
4. 在`libentry`类的`run`方法中：

```python
        for p in self.jit_function.params[len(args) :]:
            if p.name in kwargs:
                val = kwargs[p.name]
            # elif p.default is inspect._empty:
            #     continue
            else:
                val = p.default
```

我将判断参数是`inspect._empty`类的逻辑注释掉了，解决了triton参数数量不匹配的报错。尚不清楚为什么要加这个逻辑。

5. 在`randperm`, `topk`算子中，有一些最大最小常量的定义，如：

```python
_MIN_INT8_VAL:tl.constexpr = torch.iinfo(torch.int8).min
```

但是会得到报错：

```text
NameError("Cannot access global variable _MAX_FLOAT32_VAL from within @jit'ed function.
Triton kernels can only access global variables that are instanstiated as constexpr (`x = triton.language.constexpr(42)`).
Note that this is different from annotating a variable as constexpr (`x: triton.language.constexpr = 42`), which is not supported.
Alternatively, set the envvar TRITON_ALLOW_NON_CONSTEXPR_GLOBALS=1, but we do not promise to support this forever.")
```

根据该信息进行修改：

```python
_MIN_INT8_VAL = tl.constexpr(torch.iinfo(torch.int8).min)
......
```

6. `randperm`算子用到了triton jit函数`radix_type_convert`：

```python
@triton.jit
def radix_type_convert(k):
    if tl.constexpr(k.dtype == tl.int8):
        ik = k.to(tl.int8, bitcast=True)
        mask = (ik >> 7) & 0x1
        o = tl.where(mask, ik & 0x7F, ik | 0x80) # note this
    elif tl.constexpr(k.dtype == tl.int16):
        ......
    else:
        o = k
    return o
```

运行后报错，例如：

```text
        o = tl.where(mask, ik & 0x7FFFFFFF, ik | 0x80000000)
                                            ^
ValueError('Scalar 2147483648 is out of range for type int32')
```

原因是`0x80000000`被triton视作int32，而int32的最大值是`0x7FFFFFFF`，整数溢出了。
这个函数的功能其实就是要将`k`转化成无符号整数，这里取`0x80000000`，其实就是要将`k`的最高位设置为1，此时用`-0x80000000`就可以正确表示比特位且不会整数溢出了。所以，可以将`o = tl.where(..., ik | 0x80000000)`替换成`o = tl.where(..., ik | -0x80000000)`。

7. 在运行`test_accuracy_trunc_divide_scalar_scalar`时发现结果错误，检查了一下发现:

```python
def trunc_divide(A, B):
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        return trunc_div_func(A, B)
    elif isinstance(A, torch.Tensor):
        return trunc_div_func_tensor_scalar(A, B)
    ......
    else:
        return torch.tensor(A / B) # note this
```

  最后一行是`A / B`，我将其改成了`A // B`，这是本身就有的bug吗？

8. triton cpu后端和sleef都没有`isfinite`函数的支持，因此所有跟`isfnite`相关的算子与测试均无法跑通。不过该函数主要用在`isclose`和`allclose`算子中，现在都要启用CPU了，没必要额外设置`isclose`和`allclose`算子，已经在相关的测试中加入了skipif语句。

## 疑惑

- Triton中，`triton.language.math`和`triton.language.extra.xxx.libdevice`是什么关系？
- `div_rz`意思是round to zero，为什么FlagGems的`trunc_div_func`系列函数会返回`trunc(div_rz(x, y))`?

## 运行

Triton-CPU 3.2版本下，部分算子可能不出现问题，因此，请确保安装的Triton-CPU版本在3.3以上。
运行程序前，请先`export GEMS_VENDOR="arm"`，然后在triton源代码中，向`triton.language.extra.cpu.libdevice`中添加：

```python
@core.extern
def div_rn(arg0, arg1, _builder=None):
    return core.tensor(_builder.create_precise_divf(arg0.handle, arg1.handle), arg0.type)

@core.extern
def div_rz(arg0, arg1, _builder=None):
    rn = core.tensor(_builder.create_precise_divf(arg0.handle, arg1.handle), arg0.type)
    return core.tensor(_builder.create_trunc(rn.handle), rn.type)

@core.extern
def rint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("Sleef_rintf(numel)", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("Sleef_rint(numel)" , core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def atan2(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("Sleef_atan2f%(numel)_u10", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("Sleef_atan2%(numel)_u10", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)
```

`FlagGems`的`tl_extra_shim`会用到`libdevice`中的函数，如果遇到缺失，查阅`Sleef`的文档补全即可。

## TIPS

- `rint(x)`和`trunc(x)`的区别
  - `rint(x)` 向最近的偶数取整。（Round to zero, ties to even）
  - `trunc(x)`去掉小数部分。
- 上面的`rint`函数中，当`is_pure`为真，表示函数没有副作用，对于相同的输入总是能得到相同的输出，允许编译器优化；`_builder`应该是IRBuilder.

## MISC

- 如果遇到conda动态链接库版本太旧，链接到自己的动态库就行。
- src/flag_gems/runtime下的`commom_utils.py`文件名拼写错误
