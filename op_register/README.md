## `Oneflow `阅读笔记一   OP注册流程的源码分析

本文通过最少的代码复现`Oneflow OP `注册流程， 简化后的核心代码如下：

```c++
static UserOpRegisterTrigger a1 = UserOpRegistryMgr::Get().CheckAndGetOpGradRegistry("relu").SetDataTypeInferFn();
```
### 注册过程可以简化成两部分：

```c++
// 通过一些列的操作得到构造出OpRegistry test的对象。
OpRegistry test = UserOpRegistryMgr::Get().CheckAndGetOpGradRegistry("relu").SetDataTypeInferFn();

// 创建静态的全局对象a1
UserOpRegisterTrigger  a1{test}
```

- 第一部分主要构建test对象，其流程如下:

  ```c++
  UserOpRegistryMgr::Get() // 获得全局的静态对象（第一次获取的时候创建此对象）
      |
      --->CheckAndGetOpRegistry("relu")
          |
          ---->OpRegistry().Name(op_type_name) //构造一个OpRegistry对象 并对其name赋值 OpRegistry().Name(op_type_name);
         	/*
         		此过程调用的各个构造函数如下：
              OpRegistryResult init:
              OpRegistry ----------------->:
              OpRegistryResult init:
              OpRegistry Copy----------->:
              ~OpRegistry:
              ~OpRegistryResult:
   		*/
                     |
                     ---> SetDataTypeInferFn()  
  ```

  上述这一系列操作最终得到一个`OpRegistry test`的对象。

- 第二部分将test对象中的result_注册到op_reg_result\_中，其流程如下：

  ```c++
  UserOpRegistryMgr::Get()
      |
      ---> Register(test.Finish().GetResult());
   			|
              ---> op_reg_result_.emplace(result.op_type_name, result).second; 
  /*
  	此过程隐式调用函数如下：
  	UserOpRegistryMgr Register
  	~OpRegistryResult:
  	~OpRegistry:
  	~OpRegistryResult:
  */
  ```

整个过程总结为：创建`OpRegistry`类的对象 -->  初始化其成员对象`result_` -->   添加到`op_reg_result_`中。

其中`static UserOpRegisterTrigger a1` 的目的是触发上述操作。





### 实际`Oneflow`代码如下：

```c++
REGISTER_USER_OP("relu")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& in_shape = ctx->InputShape("in", 0);
      Shape* out_shape = ctx->OutputShape("out", 0);
      *out_shape = in_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      FOR_RANGE(int64_t, i, 0, in_tensor.shape().NumAxes()) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });

// 通过编译器完全展开宏
// g++ -E oneflow/user/ops/relu_op.cpp  -o relu_op.i -std=c++11 -I.  -Ibuild/python_scripts/oneflow/include/
static ::oneflow::user_op::UserOpRegisterTrigger<::oneflow::user_op::OpRegistry> g_register_trigger1 = ::oneflow::user_op::UserOpRegistryMgr::Get().CheckAndGetOpRegistry("relu")
    .Input("in")
    .Output("out")
    .SetTensorDescInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      const Shape& in_shape = ctx->InputShape("in", 0);
      Shape* out_shape = ctx->OutputShape("out", 0);
      *out_shape = in_shape;
      return Maybe<void>::Ok();
    })
    .SetGetSbpFn([](user_op::SbpContext* ctx) -> Maybe<void> {
      const user_op::TensorDesc& in_tensor = ctx->LogicalTensorDesc4InputArgNameAndIndex("in", 0);
      for (int64_t i = (0), __end = (in_tensor.shape().NumAxes()); i < __end; ++i) {
        ctx->NewBuilder()
            .Split(user_op::OpArg("in", 0), i)
            .Split(user_op::OpArg("out", 0), i)
            .Build();
      }
      return Maybe<void>::Ok();
    })
    .SetDataTypeInferFn([](user_op::InferContext* ctx) -> Maybe<void> {
      *ctx->OutputDType("out", 0) = ctx->InputDType("in", 0);
      return Maybe<void>::Ok();
    });
```

