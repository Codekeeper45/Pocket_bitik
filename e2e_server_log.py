import sys
sys.path.insert(0,'/home/hermes/integrations/serverix')
from serverix_ctl import ServerixClient
c=ServerixClient(); text=c.get_file_content('74e81a81','bot.log')
lines=text.splitlines()
for line in lines[-int(sys.argv[1] if len(sys.argv)>1 else 50):]:
 print(line)
