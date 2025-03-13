@echo off
call C:\Users\Jossie\Documents\projecten\pred_stockprice\env\Scripts\activate.bat
cd C:\Users\Jossie\Documents\projecten\pred_stockprice
python training.py AD.AS yahoo 1
call C:\Users\Jossie\Documents\projecten\pred_stockprice\env\Scripts\deactivate.bat