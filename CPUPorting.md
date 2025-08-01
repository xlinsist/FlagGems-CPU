# FlagGems CPU迁移

## 关键信息

Test统计（QUICK_MODE下, 处理器13400）

| Test | Success | Failure | Skipped | Note |
| ---- | ---- | ---- | ---- | ---- |
| attention_ops |  |  |  | 无法完成测试，进程总是被系统终止 |
| binary_pointwise_ops | 948 | 53 | 181 | 失败的测试全都精度不足 |
| blas_ops | 161 | 0 | 0 | ✔ |
| distribution_ops | 9 | 0 | 0 | ✔ |
| general_reduction_ops | 56 | 70 | 6 | 测试时间过长；全都是编译错误 |
| libentry | 5 | 0 | 1 | ✔；跳过的测试似乎是因为需要加速卡 |
| norm_ops | 58 | 0 | 2 | ✔；测试时间过长 |
| pointwise_dynamic | 65 | 0 | 14 | ✔；所跳过的测试都不是CPU的原因 |
| pointwise_dynamic_type_promotion | 22 | 0 | 3 | where算子 |
| quant | 0 | 0 | 12 | 需要CUDA |
| reduction_ops | 158 | 12 | 1 | 测试时间过长(跑了十几个小时)，规模缩小困难 |
| shape_utils | 17 | 0 | 0 | ✔ |
| special_ops | |  | | 测试时间过长，已观测到的测试大多正确 |
| tensor_constructor_ops | 339 | 0 | 0 | ✔ |
| tensor_wrapper | 4 | 0 | 0 | ✔ |
| unary_pointwise_ops | 232 | 4 | 5 | 编译错误；精度不够 |

*跑测试时，有时会出现如下报错：*

```text
ImportError while loading conftest '/home/cyanic/repos/FlagGems-CPU/tests/conftest.py'.
conftest.py:8: in <module>
    import flag_gems
../src/flag_gems/__init__.py:15: in <module>
    from .fused import *  # noqa: F403
../src/flag_gems/fused/__init__.py:1: in <module>
    from flag_gems.fused.concat_and_cache_mla import concat_and_cache_mla
../src/flag_gems/fused/concat_and_cache_mla.py:7: in <module>
    from flag_gems.utils import libentry
../src/flag_gems/utils/__init__.py:1: in <module>
    from .libentry import libentry, libtuner
../src/flag_gems/utils/libentry.py:144: in <module>
    libcache = LibCache()
../src/flag_gems/utils/libentry.py:82: in __init__
    self.preload()
../src/flag_gems/utils/libentry.py:119: in preload
    config = triton.Config(kwargs, **numargs)
E   TypeError: Config.__init__() got an unexpected keyword argument 'TILE_K'
```

目前只能通过删除`~/.flaggems`目录下的config_cache文件夹来解决这个问题，具体原因需要深入源码探寻。

另外，测试规模的进一步缩小可能会影响到原本测试的有效性，所以可以考虑为CPU测试单独添加一个CPU_QUICK_MODE。

## 运行

Triton-CPU 3.2版本下，部分算子可能会出现问题，因此，请确保安装的Triton-CPU版本在3.3以上。
运行程序前，请先`export GEMS_VENDOR="arm"`，然后在triton-cpu目录下，/third_party/cpu/language/cpu/libdevice.py中（即`triton.language.extra.cpu.libdevice`）添加：

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

## 相关PR

我将部分程序的运行截图与运行情况放在了相关PR中。

- [初版适配PR](https://github.com/xlinsist/FlagGems-CPU/pull/1)：该PR中包含部分运行测试运行截图。
- [benchmark适配PR](https://github.com/xlinsist/FlagGems-CPU/pull/3)：该PR为CPU适配了benchmark.
- [benchmark调优PR](https://github.com/xlinsist/FlagGems-CPU/pull/5)：该PR中包含调优`mm`与`addmm`算子后的运行截图。
  调优后，对于较大尺寸输入，addmm算子性能有较大提升(60%+)，mm算子有小幅提升（~20%）。但是对于小尺寸输入，由于多线程开销,性能有所下降。

## 大致修改

1. 在`src/runtime/common_utils.py`中，加入了arm:

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
    ASCEND = 9
    ARM = 10
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

## 测试详细情况

- tests: 大多数测试都能跑通，现有问题在下面的`TODO`中说明

  - [ ] attention_ops：很多算子需要加速卡；测试时间长；进程总是被系统终止；部分通过
  - [ ] binary_pointwise_ops
    - [ ] clamp, nan_to_num算子：精度不足
    - [ ] `where` related ops：`AssertionError: CPU only. There seems a mistake to dispatch to here.`
    - [x] `floor_divide`，`remainder`相关测试报错：`Floating point exception`。
          该报错与Triton的对0取余行为有关。
    - [ ] `remainder`相关测试精度不足
  - [x] blas_ops
  - [x] distribution_ops
  - [ ] general_reduction_ops
    - 测试时间需要半小时左右，单次测试时间较长，难以缩短测试时间
    - 错误全部集中在allclose算子中，为编译错误
  - [x] libentry
  - [ ] norm_ops
    - [ ] `batch_norm`：精度不足
    - [ ] 测试时间过长
  - [ ] generic_dynamic
    - [ ] benchmark目录中报错（目前版本已跳过报错示例）: `GenericBenchmark`
  - [x] pointwise_dynamic
  - [ ] pointwise_dynamic_type_promotion
    - [ ] `where` related ops：`AssertionError: CPU only. There seems a mistake to dispatch to here.`
  - [ ] reduction_ops
    - [ ] 测试时间过长，需要十几个小时
    - [ ] `batch_norm`：精度不足
    - [ ] benchmark目录中报错（目前版本已跳过报错示例）: `dot`
  - [x] shape_utils
  - [ ] special_ops
    - [ ] 精度不足: `pad`, `kron`
    - [ ] benchmark目录中报错（目前版本已跳过报错示例）: `diag`, `unique`, `isin`, `embedding`, `test_special_operations_benchmark`
  - [x] tensor_constructor
        该测试中的`test_accuracy_randn`小概率出现精度不够的问题
  - [x] tensor_wrapper
  - [ ] unary_pointwise_ops
    - [ ] angle: RuntimeError: failed to translate module to LLVM IR
    - [ ] gelu: AssertionError: Tensor-likes are not close!
    - [ ] silu: triton.compiler.errors.CompilationError
  - [ ] test_quant
        相关算子需要CUDA

## TODO

- [ ] `Philox API`：对于`distribution ops`，采用了很多`cuda philox`，目前只采用了最简单的处理。
      相关函数：multinomial, randperm, rand_like, rand, randn, randn_like
- [ ] `libtuner`：`mm`算子采用`libtuner`，但是会报错。替换成`triton.autotuner`才能正常运行。
- [ ] `benchmark`运行时间过长，调整测试规模。
- [ ] 移除分布在几十个文件中多余的`torch_device_fn`，该变量使用场景相当局限和单一，可以考虑将该全局变量优化掉。
- [ ] 优化`multinomial`算子。注：该算子存在改动且确定存在需要对CPU端进行进一步优化，因此已将其放在cpu后端的算子目录中。
- [ ] 重新实现`where`算子，该算子目前需要GPU。
- [ ] 目前对于`div`算子文件中`__remainder`函数的处理方式不一定正确，后续需要确定其正确性。同时`floordiv`相关函数也需要关注。

## TIPS

- `rint(x)`和`trunc(x)`的区别
  - `rint(x)` 向最近的偶数取整。（Round to zero, ties to even）
  - `trunc(x)`去掉小数部分。
- 上面的`rint`函数中，当`is_pure`为真，表示函数没有副作用，对于相同的输入总是能得到相同的输出，允许编译器优化；`_builder`应该是IRBuilder.

## MISC

- 如果遇到conda动态链接库版本太旧，链接到自己的动态库就行。
