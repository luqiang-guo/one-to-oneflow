#include <iostream>
#include <string>
#include <map>


class  OpRegistryResult
{
public:
    OpRegistryResult()
    {
        std::cout<< "OpRegistryResult init:"<< std::endl;
    }

    ~OpRegistryResult() 
    { 
        std::cout << "~OpRegistryResult:" << std::endl; 
    }


    std::string op_type_name;
};

class  OpRegistry
{
public:
    OpRegistry()
    {
        std::cout<< "OpRegistry ----------------->:"<< std::endl;
    }

    //Copy Construction
    OpRegistry(const OpRegistry &source)
    {
        result_ = source.result_;
        std::cout<< "OpRegistry Copy----------->:"<< std::endl;
    }

    OpRegistry& operator= (const OpRegistry & source)
    {
        result_ = source.result_;
        std::cout << "OpRegistry Deep Copy Assignment..............." << std::endl;
        return *this;
    }

    OpRegistry(OpRegistry && source)
        :result_{source.result_} //Initialize target object
    {
        
        std::cout << "OpRegistry Constructor..." << std::endl;
    }

    OpRegistry& operator= (OpRegistry && source)
    {
        result_ = source.result_;
        std::cout << "OpRegistry Move Assignment................" << std::endl;
        return *this;
    }

    ~OpRegistry() 
    { 
        std::cout << "~OpRegistry:" << std::endl; 
    }
    OpRegistry&  Finish()
    {
        std::cout<< "OpRegistry Finish"<< std::endl;
        return *this;
    }

    OpRegistry SetDataTypeInferFn()
    {
        // DataTypeInferFn data_type_infer_fn
        // result_.data_type_infer_fn = std::move(data_type_infer_fn);
        std::cout<< "___________SetDataTypeInferFn_____________"<< std::endl;
        return *this;
    }
    OpRegistry& Name(const std::string& op_type_name) 
    {
        result_.op_type_name = op_type_name;
        return *this;
    }

    OpRegistryResult GetResult() { return result_; }

    OpRegistryResult result_;
};

class UserOpRegistryMgr final
{
public :

    UserOpRegistryMgr()
    {
        std::cout<< "UserOpRegistryMgr init:"<< std::endl;
    }

    ~UserOpRegistryMgr() 
    { 
        std::cout << "~UserOpRegistryMgr:" << std::endl; 
    }

    static UserOpRegistryMgr& Get()
    {
        std::cout << "Get   -------->" << std::endl; 
        static UserOpRegistryMgr mgr;
        return mgr;
    }

    OpRegistry CheckAndGetOpGradRegistry(const std::string& op_type_name)
    {
        std::cout << "CheckAndGetOpGradRegistry  function -------->" << std::endl; 
        return OpRegistry().Name(op_type_name);
    }

    void Register(OpRegistryResult result)
    {
        std::cout<< "UserOpRegistryMgr Register"<< std::endl;
        op_reg_result_.emplace(result.op_type_name, result).second;
    }


    std::map<std::string, OpRegistryResult> op_reg_result_;
};


struct UserOpRegisterTrigger final
{
    UserOpRegisterTrigger(OpRegistry registry)
    {
        std::cout << "~~~~~~~~~~~UserOpRegisterTrigger~~~~~~~~~~" << std::endl;
        UserOpRegistryMgr::Get().Register(registry.Finish().GetResult());
    }

    ~UserOpRegisterTrigger()
    {
        std::cout << "~~~~~~~~~~~~~~~~~~~~~" << std::endl;
    }
};

static UserOpRegisterTrigger a1 = UserOpRegistryMgr::Get().CheckAndGetOpGradRegistry("relu").SetDataTypeInferFn();
// static UserOpRegisterTrigger a2 = UserOpRegistryMgr::Get().CheckAndGetOpGradRegistry("prelu").SetDataTypeInferFn();

int main()
{
    // for(const auto& [key, value] : op_reg_result_) { std::cout << key << " " << std::endl; }
    for (auto iter = UserOpRegistryMgr::Get().op_reg_result_.begin(); iter !=UserOpRegistryMgr::Get().op_reg_result_.end(); iter++ ) 
    {
        std::cout << iter->first << " ____iter______ " << std::endl;
    }
    std::cout << "return ------->" << std::endl;
    return 0;
}