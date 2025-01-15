import sys

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        _,_,exc_tb=error_detail.exc_info()
        line_number=exc_tb.tb_lineno
        file_name=exc_tb.tb_frame.f_code.co_filename
        self.error_message="Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(
            file_name,line_number,str(error_message))
    
    def __str__(self):
        return self.error_message
    


        