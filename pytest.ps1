$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir\pytest.py" @args
