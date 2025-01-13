class ModelNotTrainedError(RuntimeError):
    """
    模型未拟合就使用会触发此错误。
    """
    def __init__(self, message=None, additional_info=None):
        """
        - param message: 错误的描述性消息。
        - param additional_info: 任何额外的错误相关信息。
        """
        super().__init__(message)
        self.additional_info = additional_info
 
    def __str__(self):
        """
        返回错误的额外信息字符串表示形式。
        """
        base_message = super().__str__()
        if self.additional_info:
            return f"{base_message} (Additional info: {self.additional_info})"
        return base_message
    
class ParamatersError(ValueError):
    """
    参数传递不正确就会触发此错误
    """
    def __init__(self, message=None, additional_info=None):
        """
        - param message: 错误的描述性消息。
        - param additional_info: 任何额外的错误相关信息。
        """
        super().__init__(message)
        self.additional_info = additional_info
 
    def __str__(self):
        """
        返回错误的额外信息字符串表示形式。
        """
        base_message = super().__str__()
        if self.additional_info:
            return f"{base_message} (Additional info: {self.additional_info})"
        return base_message