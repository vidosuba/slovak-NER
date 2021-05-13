import trankit
from trankit import Pipeline

# trankit.verify_customized_pipeline(
#     category='customized-ner', # pipeline category
#     save_dir='./output_dir' # directory used for saving models in previous steps
# )

p = Pipeline(lang='customized-ner', cache_dir='./output_dir')
#res = p.ner('''Vlani súd v Bratislave vymeral Marošovi Deákovi 25 rokov väzenia za rôzne zločiny''', is_sent=True)
res = p.ner('''Vláda už minula miliardovú rezervu v rozpočte, ktorá bola určená na krytie výdavkov súvisiacich s pandémiou. 
Minister financií a predseda OĽaNO Igor Matovič preto predložil návrh na ďalšie zvýšenie výdavkov rozpočtu.
Konkrétne rezerva v rozpočte sa má zvýšiť o 2,4 miliardy eur a nepandemické výdavky sa majú zvýšiť o 984 miliónov. 
Napríklad časť peňazí sa má použiť na dofinancovanie železničných spoločností, na výdavky súvisiace s plánom obnovy či na vyššie odvody do rozpočtu EÚ.
Dokopy je to nárast rozpočtových výdavkov v tomto roku o 3,4 miliardy.
Návrh vláda schválila s pripomienkami, problém so zvyšovaním výdavkov majú v SaS. 
„Takto sa nedá pristupovať k verejným peniazom, z večera do rána predložiť nejaký dokument a rýchlo schváliť, a nepýtať sa,“ povedal minister hospodárstva Richard Sulík. 
Návrh nepodporil ani jeho stranícky kolega a minister školstva Branislav Gröhling.
Sulík povedal, že proti by hlasoval aj minister zahraničných vecí Ivan Korčok, ak by bol na rokovaní vlády prítomný. 
„Toto je proti DNA SaS, aby sme tu len takto zvýšili výdavky o 3,5 miliardy eur. Je to vec, ktorú chceme prediskutovať, máme tam mnoho otázok ku konkrétnym položkám.“ 
Žiada preto riadnu diskusiu a odôvodnenie.
Nejde však o novú informáciu, že deficit sa má v tomto roku zvýšiť na takmer 10 %. Hovorí sa o tom aj v pláne obnovy, ktorý vláda pred dvoma týždňami poslala do Bruselu: 
„V aktuálnom roku sa pri zohľadnení dodatočnej rezervy na krytie vplyvov pandémie až na úrovni 2 % HDP uvažuje s navýšením deficitu na 9,9 % HDP.“''', is_sent=True)
print(res)