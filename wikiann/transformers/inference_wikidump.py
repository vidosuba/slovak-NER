from transformers import pipeline
from spacy import displacy
import nltk
import jsonlines
import textspan
from tqdm import tqdm


nltk.download('punkt')

# annotations = [{'sentences': [{'text': 'Eswatini , dlhý tvar Eswatinské kráľovstvo ( do roku 2018 [ na Slovensku : 2019 ] : Svazijsko , dlhý tvar Svazijské kráľovstvo ) je štát v Afrike .', 'ents': [{'text': 'Eswatinské kráľovstvo', 'label': 'LOC', 'start': 21, 'end': 42}, {'text': 'Slovensku', 'label': 'LOC', 'start': 63, 'end': 72}, {'text': 'Svazijské kráľovstvo', 'label': 'LOC', 'start': 106, 'end': 126}, {'text': 'Afrike', 'label': 'LOC', 'start': 139, 'end': 145}]},
#                {'text': 'Hlavné mesto je Mbabane .', 'ents': [{'text': 'Hlavné mesto', 'label': 'LOC', 'start': 0, 'end': 12}, {'text': 'babane', 'label': 'LOC', 'start': 17, 'end': 23}]}],
#                'meta': {'id': '0002'}
#                }]

#annotations_prodigy = {"text":"Eswatini , dlh\u00fd tvar Eswatinsk\u00e9 kr\u00e1\u013eovstvo ( do roku 2018 [ na Slovensku : 2019 ] : Svazijsko , dlh\u00fd tvar Svazijsk\u00e9 kr\u00e1\u013eovstvo ) je \u0161t\u00e1t v Afrike .\n\tHlavn\u00e9 mesto je Mbabane .\n\tSvazijsko je od roku 1968 nez\u00e1vislou monarchiou v r\u00e1mci Britsk\u00e9ho spolo\u010denstva .\n\tP\u00f4vodne bolo \u00fazemie Svazijska ob\u00fdvan\u00e9 kme\u0148mi hovoriacimi jazykom soto .\n\tZa\u010diatkom 19. storo\u010dia dobyli \u00fazemie svazijsk\u00e9 kmene , ktor\u00e9 nesk\u00f4r Zuluovia vytla\u010dili na sever .\n\tPo\u010das vl\u00e1dy kr\u00e1\u013ea Sobhuza I. sa podarilo odrazi\u0165 Zuluov , ale nie Eur\u00f3panov .\n\tOd roku 1850 o nadvl\u00e1du v Svazijsku striedavo bojovala Ve\u013ek\u00e1 Brit\u00e1nia a b\u00farske republiky .\n\tV roku 1895 sa Svazijsko dostalo pod nadvl\u00e1du Transvaalu .\n\tPo b\u00farskej vojne bolo Svazijsko od roku 1902 pod spr\u00e1vou Ve\u013ekej Brit\u00e1nie a v roku 1906 z\u00edskalo \u0161tat\u00fat protektor\u00e1tu .\n\tPo colnej \u00fanii z roku 1910 sa Svazijsko stalo politicky a hospod\u00e1rsky nez\u00e1visl\u00fdm od Ju\u017enej Afriky .\n\tPo\u010das vl\u00e1dy kr\u00e1\u013ea Sobhuza II .\n\tz\u00edskala krajina v roku 1968 samostatnos\u0165 , v roku 1973 sa zmenou \u00fastavy posilnilo postavenie kr\u00e1\u013ea .\n\tPo jeho smrti v roku 1982 nast\u00fapil na tr\u00f3n jeden\u00e1s\u0165ro\u010dn\u00fd princ Makhosimvel .\n\tVl\u00e1dnu\u0165 za\u010dal v roku 1986 ako kr\u00e1\u013e Mswati III .\n\tDo jeho plnoletosti ( 1982 \u2013 1986 ) vl\u00e1dla regentka kr\u00e1\u013eovn\u00e1-matka .\n\tV ilegalite p\u00f4sobia r\u00f4zne politick\u00e9 zoskupenia proti vl\u00e1de .\n\tPovrch tvor\u00ed n\u00e1horn\u00e1 plo\u0161ina s najvy\u0161\u0161ou horou \u0161t\u00e1tu Emlembe zva\u017euj\u00facou sa od z\u00e1padu k v\u00fdchodu a pr\u00edkre spadaj\u00facou k Mozambickej n\u00ed\u017eine .\n\tNajv\u00e4\u010d\u0161\u00edmi riekami s\u00fa Komati a Usutu .\n\tPodnebie je subtropick\u00e9 , pomerne vlhk\u00e9 a tepl\u00e9 .\n\tZr\u00e1\u017eky prich\u00e1dzaj\u00fa predov\u0161etk\u00fdm v lete .\n\tRo\u010dne ich spadne medzi 500 a\u017e .\n\tHlavn\u00fdm a z\u00e1rove\u0148 najv\u00e4\u010d\u0161\u00edm mestom v krajine je Mbabane s viac ne\u017e 50 000 obyvate\u013emi .\n\t\u010eal\u0161\u00edmi ve\u013ek\u00fdmi mestami s\u00fa Manzini , Lobamba a Siteki .\n\tPri hraniciach s Mozambikom s\u00fa pohoria a na severoz\u00e1pade krajiny je da\u017e\u010fov\u00fd prales .\n\tZvy\u0161ok krajiny pokr\u00fdvaj\u00fa savany .\n\tSvazijsko sa del\u00ed na \u0161tyri provincie : V \u010dele krajiny stoj\u00ed kr\u00e1\u013e .\n\tOd roku 1986 to je Mswati III. , ktor\u00fd nahradil svojho predchodcu Sobhuzu II .\n\tKe\u010f sa v roku 1986 stal kr\u00e1\u013eom Mswati III. , le\u017eala pred n\u00edm krajina rozdelen\u00e1 medzi belo\u0161sk\u00fa men\u0161inu , ktor\u00e1 si chcela podr\u017ea\u0165 nadvl\u00e1du , a \u010derno\u0161sk\u00fa v\u00e4\u010d\u0161inu , ktor\u00e1 chcela roz\u0161\u00edri\u0165 svoje pr\u00e1va .\n\tBud\u00facnos\u0165 Svazijska zna\u010dne ovplyvnia historick\u00e9 zmeny odohr\u00e1vaj\u00face sa za hranicami Juhoafrickej republiky , ktorej \u013eud si roku 1994 zvolil svojho prv\u00e9ho \u010derno\u0161sk\u00e9ho prezidenta Nelsona Mandelu .\n\tNapriek tomu \u017ee ofici\u00e1lne existuje sen\u00e1t o po\u010dte 30 \u010dlenov , Svazijsko je absolutistickou diarchiou , kde spolo\u010dn\u00fdmi hlavami \u0161t\u00e1tu s\u00fa pod\u013ea trad\u00edcie kr\u00e1\u013e a jeho matka .\n\tSvazijsko je rozvojov\u00fd \u0161t\u00e1t s prevahou po\u013enohospod\u00e1rstva .\n\tZ nerastn\u00fdch surov\u00edn sa \u0165a\u017e\u00ed \u010dierne uhlie , nasleduj\u00fa diamanty , \u017eelezn\u00e1 ruda , zlato , c\u00edn a azbest .\n\tV\u00fdznam m\u00e1 v\u00fdroba cukru a produkcia dreva .\n\tPestuje sa cukrov\u00e1 trstina , tabak , kukurica , ry\u017ea , bavlna a citrusy .\n\tChov\u00e1 sa hov\u00e4dz\u00ed dobytok , kozy a hydina .\n\tVe\u013ek\u00e1 \u010das\u0165 Svazij\u010danov pracuje v zlatonosn\u00fdch baniach za hranicami v Juhoafrickej republike .","spans":[{"start":0,"end":8,"token_start":0,"token_end":0,"label":"LOC"},{"text":"Eswatinsk\u00e9 kr\u00e1\u013eovstvo","start":21,"end":42,"token_start":4,"token_end":5,"label":"LOC"},{"text":"Slovensku","start":63,"end":72,"token_start":12,"token_end":12,"label":"LOC"},{"start":84,"end":93,"token_start":17,"token_end":17,"label":"LOC"},{"text":"Svazijsk\u00e9 kr\u00e1\u013eovstvo","start":106,"end":126,"token_start":21,"token_end":22,"label":"LOC"},{"text":"Afrike","start":139,"end":145,"token_start":27,"token_end":27,"label":"LOC"},{"start":165,"end":172,"token_start":33,"token_end":33,"label":"LOC"},{"start":176,"end":185,"token_start":36,"token_end":36,"label":"LOC"},{"text":"Britsk\u00e9ho spolo\u010denstva","start":232,"end":254,"token_start":45,"token_end":46,"label":"LOC"},{"text":"Svazijska","start":278,"end":287,"token_start":52,"token_end":52,"label":"LOC"},{"text":"Zulu","start":399,"end":407,"token_start":70,"token_end":70,"label":"ORG"},{"text":"Sobhuza I.","start":448,"end":458,"token_start":79,"token_end":80,"label":"PER"},{"start":479,"end":485,"token_start":84,"token_end":84,"label":"MISC"},{"start":496,"end":505,"token_start":88,"token_end":88,"label":"MISC"},{"text":"Svazijsku","start":535,"end":544,"token_start":97,"token_end":97,"label":"LOC"},{"text":"Ve\u013ek\u00e1 Brit\u00e1nia","start":564,"end":578,"token_start":100,"token_end":101,"label":"LOC"},{"start":616,"end":625,"token_start":111,"token_end":111,"label":"LOC"},{"text":"Transvaalu","start":647,"end":657,"token_start":115,"token_end":115,"label":"ORG"},{"text":"Svazijsko","start":683,"end":692,"token_start":122,"token_end":122,"label":"LOC"},{"text":"Ve\u013ekej Brit\u00e1nie","start":718,"end":733,"token_start":128,"token_end":129,"label":"LOC"},{"start":809,"end":818,"token_start":146,"token_end":146,"label":"LOC"},{"text":"Ju\u017enej Afriky","start":863,"end":876,"token_start":153,"token_end":154,"label":"LOC"},{"text":"Sobhuza II","start":898,"end":908,"token_start":160,"token_end":161,"label":"PER"},{"text":"Makhosimvel","start":1077,"end":1088,"token_start":193,"token_end":193,"label":"PER"},{"text":"Mswati III .","start":1127,"end":1139,"token_start":203,"token_end":205,"label":"PER"},{"text":"Emlembe","start":1326,"end":1333,"token_start":238,"token_end":238,"label":"LOC"},{"text":"Mozambickej n\u00ed\u017eine","start":1390,"end":1408,"token_start":249,"token_end":250,"label":"LOC"},{"text":"Komati","start":1434,"end":1440,"token_start":256,"token_end":256,"label":"LOC"},{"text":"Usutu","start":1443,"end":1448,"token_start":258,"token_end":258,"label":"LOC"},{"text":"Mbabane","start":1626,"end":1633,"token_start":294,"token_end":294,"label":"LOC"},{"text":"Manzini","start":1693,"end":1700,"token_start":307,"token_end":307,"label":"LOC"},{"text":"Lobamba","start":1703,"end":1710,"token_start":309,"token_end":309,"label":"LOC"},{"text":"Siteki","start":1713,"end":1719,"token_start":311,"token_end":311,"label":"LOC"},{"text":"Mozambikom","start":1740,"end":1750,"token_start":317,"token_end":317,"label":"LOC"},{"text":"Svazijsko","start":1844,"end":1853,"token_start":335,"token_end":335,"label":"LOC"},{"text":"Mswati III.","start":1931,"end":1942,"token_start":354,"token_end":355,"label":"PER"},{"text":"Sobhuzu II .","start":1978,"end":1990,"token_start":361,"token_end":363,"label":"PER"},{"text":"Mswati III.","start":2023,"end":2034,"token_start":372,"token_end":373,"label":"PER"},{"start":2200,"end":2209,"token_start":402,"token_end":402,"label":"LOC"},{"text":"Juhoafrickej republiky","start":2273,"end":2295,"token_start":411,"token_end":412,"label":"LOC"},{"text":"Nelsona Mandelu","start":2366,"end":2381,"token_start":424,"token_end":425,"label":"PER"},{"text":"Svazijsko","start":2446,"end":2455,"token_start":439,"token_end":439,"label":"LOC"},{"text":"Svazijsko","start":2555,"end":2564,"token_start":457,"token_end":457,"label":"LOC"},{"start":2893,"end":2904,"token_start":524,"token_end":524,"label":"MISC"},{"text":"Juhoafrickej republike","start":2951,"end":2973,"token_start":532,"token_end":533,"label":"LOC"}],"tokens":[{"id":0,"text":"Eswatini","start":0,"end":8},{"id":1,"text":",","start":9,"end":10},{"id":2,"text":"dlh\u00fd","start":11,"end":15},{"id":3,"text":"tvar","start":16,"end":20},{"id":4,"text":"Eswatinsk\u00e9","start":21,"end":31},{"id":5,"text":"kr\u00e1\u013eovstvo","start":32,"end":42},{"id":6,"text":"(","start":43,"end":44},{"id":7,"text":"do","start":45,"end":47},{"id":8,"text":"roku","start":48,"end":52},{"id":9,"text":"2018","start":53,"end":57},{"id":10,"text":"[","start":58,"end":59},{"id":11,"text":"na","start":60,"end":62},{"id":12,"text":"Slovensku","start":63,"end":72},{"id":13,"text":":","start":73,"end":74},{"id":14,"text":"2019","start":75,"end":79},{"id":15,"text":"]","start":80,"end":81},{"id":16,"text":":","start":82,"end":83},{"id":17,"text":"Svazijsko","start":84,"end":93},{"id":18,"text":",","start":94,"end":95},{"id":19,"text":"dlh\u00fd","start":96,"end":100},{"id":20,"text":"tvar","start":101,"end":105},{"id":21,"text":"Svazijsk\u00e9","start":106,"end":115},{"id":22,"text":"kr\u00e1\u013eovstvo","start":116,"end":126},{"id":23,"text":")","start":127,"end":128},{"id":24,"text":"je","start":129,"end":131},{"id":25,"text":"\u0161t\u00e1t","start":132,"end":136},{"id":26,"text":"v","start":137,"end":138},{"id":27,"text":"Afrike","start":139,"end":145},{"id":28,"text":".","start":146,"end":147},{"id":29,"text":"\n\t","start":0,"end":2},{"id":30,"text":"Hlavn\u00e9","start":149,"end":155},{"id":31,"text":"mesto","start":156,"end":161},{"id":32,"text":"je","start":162,"end":164},{"id":33,"text":"Mbabane","start":165,"end":172},{"id":34,"text":".","start":173,"end":174},{"id":35,"text":"\n\t","start":149,"end":151},{"id":36,"text":"Svazijsko","start":176,"end":185},{"id":37,"text":"je","start":186,"end":188},{"id":38,"text":"od","start":189,"end":191},{"id":39,"text":"roku","start":192,"end":196},{"id":40,"text":"1968","start":197,"end":201},{"id":41,"text":"nez\u00e1vislou","start":202,"end":212},{"id":42,"text":"monarchiou","start":213,"end":223},{"id":43,"text":"v","start":224,"end":225},{"id":44,"text":"r\u00e1mci","start":226,"end":231},{"id":45,"text":"Britsk\u00e9ho","start":232,"end":241},{"id":46,"text":"spolo\u010denstva","start":242,"end":254},{"id":47,"text":".","start":255,"end":256},{"id":48,"text":"\n\t","start":176,"end":178},{"id":49,"text":"P\u00f4vodne","start":258,"end":265},{"id":50,"text":"bolo","start":266,"end":270},{"id":51,"text":"\u00fazemie","start":271,"end":277},{"id":52,"text":"Svazijska","start":278,"end":287},{"id":53,"text":"ob\u00fdvan\u00e9","start":288,"end":295},{"id":54,"text":"kme\u0148mi","start":296,"end":302},{"id":55,"text":"hovoriacimi","start":303,"end":314},{"id":56,"text":"jazykom","start":315,"end":322},{"id":57,"text":"soto","start":323,"end":327},{"id":58,"text":".","start":328,"end":329},{"id":59,"text":"\n\t","start":258,"end":260},{"id":60,"text":"Za\u010diatkom","start":331,"end":340},{"id":61,"text":"19.","start":341,"end":344},{"id":62,"text":"storo\u010dia","start":345,"end":353},{"id":63,"text":"dobyli","start":354,"end":360},{"id":64,"text":"\u00fazemie","start":361,"end":367},{"id":65,"text":"svazijsk\u00e9","start":368,"end":377},{"id":66,"text":"kmene","start":378,"end":383},{"id":67,"text":",","start":384,"end":385},{"id":68,"text":"ktor\u00e9","start":386,"end":391},{"id":69,"text":"nesk\u00f4r","start":392,"end":398},{"id":70,"text":"Zuluovia","start":399,"end":407},{"id":71,"text":"vytla\u010dili","start":408,"end":417},{"id":72,"text":"na","start":418,"end":420},{"id":73,"text":"sever","start":421,"end":426},{"id":74,"text":".","start":427,"end":428},{"id":75,"text":"\n\t","start":331,"end":333},{"id":76,"text":"Po\u010das","start":430,"end":435},{"id":77,"text":"vl\u00e1dy","start":436,"end":441},{"id":78,"text":"kr\u00e1\u013ea","start":442,"end":447},{"id":79,"text":"Sobhuza","start":448,"end":455},{"id":80,"text":"I.","start":456,"end":458},{"id":81,"text":"sa","start":459,"end":461},{"id":82,"text":"podarilo","start":462,"end":470},{"id":83,"text":"odrazi\u0165","start":471,"end":478},{"id":84,"text":"Zuluov","start":479,"end":485},{"id":85,"text":",","start":486,"end":487},{"id":86,"text":"ale","start":488,"end":491},{"id":87,"text":"nie","start":492,"end":495},{"id":88,"text":"Eur\u00f3panov","start":496,"end":505},{"id":89,"text":".","start":506,"end":507},{"id":90,"text":"\n\t","start":430,"end":432},{"id":91,"text":"Od","start":509,"end":511},{"id":92,"text":"roku","start":512,"end":516},{"id":93,"text":"1850","start":517,"end":521},{"id":94,"text":"o","start":522,"end":523},{"id":95,"text":"nadvl\u00e1du","start":524,"end":532},{"id":96,"text":"v","start":533,"end":534},{"id":97,"text":"Svazijsku","start":535,"end":544},{"id":98,"text":"striedavo","start":545,"end":554},{"id":99,"text":"bojovala","start":555,"end":563},{"id":100,"text":"Ve\u013ek\u00e1","start":564,"end":569},{"id":101,"text":"Brit\u00e1nia","start":570,"end":578},{"id":102,"text":"a","start":579,"end":580},{"id":103,"text":"b\u00farske","start":581,"end":587},{"id":104,"text":"republiky","start":588,"end":597},{"id":105,"text":".","start":598,"end":599},{"id":106,"text":"\n\t","start":509,"end":511},{"id":107,"text":"V","start":601,"end":602},{"id":108,"text":"roku","start":603,"end":607},{"id":109,"text":"1895","start":608,"end":612},{"id":110,"text":"sa","start":613,"end":615},{"id":111,"text":"Svazijsko","start":616,"end":625},{"id":112,"text":"dostalo","start":626,"end":633},{"id":113,"text":"pod","start":634,"end":637},{"id":114,"text":"nadvl\u00e1du","start":638,"end":646},{"id":115,"text":"Transvaalu","start":647,"end":657},{"id":116,"text":".","start":658,"end":659},{"id":117,"text":"\n\t","start":601,"end":603},{"id":118,"text":"Po","start":661,"end":663},{"id":119,"text":"b\u00farskej","start":664,"end":671},{"id":120,"text":"vojne","start":672,"end":677},{"id":121,"text":"bolo","start":678,"end":682},{"id":122,"text":"Svazijsko","start":683,"end":692},{"id":123,"text":"od","start":693,"end":695},{"id":124,"text":"roku","start":696,"end":700},{"id":125,"text":"1902","start":701,"end":705},{"id":126,"text":"pod","start":706,"end":709},{"id":127,"text":"spr\u00e1vou","start":710,"end":717},{"id":128,"text":"Ve\u013ekej","start":718,"end":724},{"id":129,"text":"Brit\u00e1nie","start":725,"end":733},{"id":130,"text":"a","start":734,"end":735},{"id":131,"text":"v","start":736,"end":737},{"id":132,"text":"roku","start":738,"end":742},{"id":133,"text":"1906","start":743,"end":747},{"id":134,"text":"z\u00edskalo","start":748,"end":755},{"id":135,"text":"\u0161tat\u00fat","start":756,"end":762},{"id":136,"text":"protektor\u00e1tu","start":763,"end":775},{"id":137,"text":".","start":776,"end":777},{"id":138,"text":"\n\t","start":661,"end":663},{"id":139,"text":"Po","start":779,"end":781},{"id":140,"text":"colnej","start":782,"end":788},{"id":141,"text":"\u00fanii","start":789,"end":793},{"id":142,"text":"z","start":794,"end":795},{"id":143,"text":"roku","start":796,"end":800},{"id":144,"text":"1910","start":801,"end":805},{"id":145,"text":"sa","start":806,"end":808},{"id":146,"text":"Svazijsko","start":809,"end":818},{"id":147,"text":"stalo","start":819,"end":824},{"id":148,"text":"politicky","start":825,"end":834},{"id":149,"text":"a","start":835,"end":836},{"id":150,"text":"hospod\u00e1rsky","start":837,"end":848},{"id":151,"text":"nez\u00e1visl\u00fdm","start":849,"end":859},{"id":152,"text":"od","start":860,"end":862},{"id":153,"text":"Ju\u017enej","start":863,"end":869},{"id":154,"text":"Afriky","start":870,"end":876},{"id":155,"text":".","start":877,"end":878},{"id":156,"text":"\n\t","start":779,"end":781},{"id":157,"text":"Po\u010das","start":880,"end":885},{"id":158,"text":"vl\u00e1dy","start":886,"end":891},{"id":159,"text":"kr\u00e1\u013ea","start":892,"end":897},{"id":160,"text":"Sobhuza","start":898,"end":905},{"id":161,"text":"II","start":906,"end":908},{"id":162,"text":".","start":909,"end":910},{"id":163,"text":"\n\t","start":880,"end":882},{"id":164,"text":"z\u00edskala","start":912,"end":919},{"id":165,"text":"krajina","start":920,"end":927},{"id":166,"text":"v","start":928,"end":929},{"id":167,"text":"roku","start":930,"end":934},{"id":168,"text":"1968","start":935,"end":939},{"id":169,"text":"samostatnos\u0165","start":940,"end":952},{"id":170,"text":",","start":953,"end":954},{"id":171,"text":"v","start":955,"end":956},{"id":172,"text":"roku","start":957,"end":961},{"id":173,"text":"1973","start":962,"end":966},{"id":174,"text":"sa","start":967,"end":969},{"id":175,"text":"zmenou","start":970,"end":976},{"id":176,"text":"\u00fastavy","start":977,"end":983},{"id":177,"text":"posilnilo","start":984,"end":993},{"id":178,"text":"postavenie","start":994,"end":1004},{"id":179,"text":"kr\u00e1\u013ea","start":1005,"end":1010},{"id":180,"text":".","start":1011,"end":1012},{"id":181,"text":"\n\t","start":912,"end":914},{"id":182,"text":"Po","start":1014,"end":1016},{"id":183,"text":"jeho","start":1017,"end":1021},{"id":184,"text":"smrti","start":1022,"end":1027},{"id":185,"text":"v","start":1028,"end":1029},{"id":186,"text":"roku","start":1030,"end":1034},{"id":187,"text":"1982","start":1035,"end":1039},{"id":188,"text":"nast\u00fapil","start":1040,"end":1048},{"id":189,"text":"na","start":1049,"end":1051},{"id":190,"text":"tr\u00f3n","start":1052,"end":1056},{"id":191,"text":"jeden\u00e1s\u0165ro\u010dn\u00fd","start":1057,"end":1070},{"id":192,"text":"princ","start":1071,"end":1076},{"id":193,"text":"Makhosimvel","start":1077,"end":1088},{"id":194,"text":".","start":1089,"end":1090},{"id":195,"text":"\n\t","start":1014,"end":1016},{"id":196,"text":"Vl\u00e1dnu\u0165","start":1092,"end":1099},{"id":197,"text":"za\u010dal","start":1100,"end":1105},{"id":198,"text":"v","start":1106,"end":1107},{"id":199,"text":"roku","start":1108,"end":1112},{"id":200,"text":"1986","start":1113,"end":1117},{"id":201,"text":"ako","start":1118,"end":1121},{"id":202,"text":"kr\u00e1\u013e","start":1122,"end":1126},{"id":203,"text":"Mswati","start":1127,"end":1133},{"id":204,"text":"III","start":1134,"end":1137},{"id":205,"text":".","start":1138,"end":1139},{"id":206,"text":"\n\t","start":1092,"end":1094},{"id":207,"text":"Do","start":1141,"end":1143},{"id":208,"text":"jeho","start":1144,"end":1148},{"id":209,"text":"plnoletosti","start":1149,"end":1160},{"id":210,"text":"(","start":1161,"end":1162},{"id":211,"text":"1982","start":1163,"end":1167},{"id":212,"text":"\u2013","start":1168,"end":1169},{"id":213,"text":"1986","start":1170,"end":1174},{"id":214,"text":")","start":1175,"end":1176},{"id":215,"text":"vl\u00e1dla","start":1177,"end":1183},{"id":216,"text":"regentka","start":1184,"end":1192},{"id":217,"text":"kr\u00e1\u013eovn\u00e1-matka","start":1193,"end":1207},{"id":218,"text":".","start":1208,"end":1209},{"id":219,"text":"\n\t","start":1141,"end":1143},{"id":220,"text":"V","start":1211,"end":1212},{"id":221,"text":"ilegalite","start":1213,"end":1222},{"id":222,"text":"p\u00f4sobia","start":1223,"end":1230},{"id":223,"text":"r\u00f4zne","start":1231,"end":1236},{"id":224,"text":"politick\u00e9","start":1237,"end":1246},{"id":225,"text":"zoskupenia","start":1247,"end":1257},{"id":226,"text":"proti","start":1258,"end":1263},{"id":227,"text":"vl\u00e1de","start":1264,"end":1269},{"id":228,"text":".","start":1270,"end":1271},{"id":229,"text":"\n\t","start":1211,"end":1213},{"id":230,"text":"Povrch","start":1273,"end":1279},{"id":231,"text":"tvor\u00ed","start":1280,"end":1285},{"id":232,"text":"n\u00e1horn\u00e1","start":1286,"end":1293},{"id":233,"text":"plo\u0161ina","start":1294,"end":1301},{"id":234,"text":"s","start":1302,"end":1303},{"id":235,"text":"najvy\u0161\u0161ou","start":1304,"end":1313},{"id":236,"text":"horou","start":1314,"end":1319},{"id":237,"text":"\u0161t\u00e1tu","start":1320,"end":1325},{"id":238,"text":"Emlembe","start":1326,"end":1333},{"id":239,"text":"zva\u017euj\u00facou","start":1334,"end":1344},{"id":240,"text":"sa","start":1345,"end":1347},{"id":241,"text":"od","start":1348,"end":1350},{"id":242,"text":"z\u00e1padu","start":1351,"end":1357},{"id":243,"text":"k","start":1358,"end":1359},{"id":244,"text":"v\u00fdchodu","start":1360,"end":1367},{"id":245,"text":"a","start":1368,"end":1369},{"id":246,"text":"pr\u00edkre","start":1370,"end":1376},{"id":247,"text":"spadaj\u00facou","start":1377,"end":1387},{"id":248,"text":"k","start":1388,"end":1389},{"id":249,"text":"Mozambickej","start":1390,"end":1401},{"id":250,"text":"n\u00ed\u017eine","start":1402,"end":1408},{"id":251,"text":".","start":1409,"end":1410},{"id":252,"text":"\n\t","start":1273,"end":1275},{"id":253,"text":"Najv\u00e4\u010d\u0161\u00edmi","start":1412,"end":1422},{"id":254,"text":"riekami","start":1423,"end":1430},{"id":255,"text":"s\u00fa","start":1431,"end":1433},{"id":256,"text":"Komati","start":1434,"end":1440},{"id":257,"text":"a","start":1441,"end":1442},{"id":258,"text":"Usutu","start":1443,"end":1448},{"id":259,"text":".","start":1449,"end":1450},{"id":260,"text":"\n\t","start":1412,"end":1414},{"id":261,"text":"Podnebie","start":1452,"end":1460},{"id":262,"text":"je","start":1461,"end":1463},{"id":263,"text":"subtropick\u00e9","start":1464,"end":1475},{"id":264,"text":",","start":1476,"end":1477},{"id":265,"text":"pomerne","start":1478,"end":1485},{"id":266,"text":"vlhk\u00e9","start":1486,"end":1491},{"id":267,"text":"a","start":1492,"end":1493},{"id":268,"text":"tepl\u00e9","start":1494,"end":1499},{"id":269,"text":".","start":1500,"end":1501},{"id":270,"text":"\n\t","start":1452,"end":1454},{"id":271,"text":"Zr\u00e1\u017eky","start":1503,"end":1509},{"id":272,"text":"prich\u00e1dzaj\u00fa","start":1510,"end":1521},{"id":273,"text":"predov\u0161etk\u00fdm","start":1522,"end":1534},{"id":274,"text":"v","start":1535,"end":1536},{"id":275,"text":"lete","start":1537,"end":1541},{"id":276,"text":".","start":1542,"end":1543},{"id":277,"text":"\n\t","start":1503,"end":1505},{"id":278,"text":"Ro\u010dne","start":1545,"end":1550},{"id":279,"text":"ich","start":1551,"end":1554},{"id":280,"text":"spadne","start":1555,"end":1561},{"id":281,"text":"medzi","start":1562,"end":1567},{"id":282,"text":"500","start":1568,"end":1571},{"id":283,"text":"a\u017e","start":1572,"end":1574},{"id":284,"text":".","start":1575,"end":1576},{"id":285,"text":"\n\t","start":1545,"end":1547},{"id":286,"text":"Hlavn\u00fdm","start":1578,"end":1585},{"id":287,"text":"a","start":1586,"end":1587},{"id":288,"text":"z\u00e1rove\u0148","start":1588,"end":1595},{"id":289,"text":"najv\u00e4\u010d\u0161\u00edm","start":1596,"end":1605},{"id":290,"text":"mestom","start":1606,"end":1612},{"id":291,"text":"v","start":1613,"end":1614},{"id":292,"text":"krajine","start":1615,"end":1622},{"id":293,"text":"je","start":1623,"end":1625},{"id":294,"text":"Mbabane","start":1626,"end":1633},{"id":295,"text":"s","start":1634,"end":1635},{"id":296,"text":"viac","start":1636,"end":1640},{"id":297,"text":"ne\u017e","start":1641,"end":1644},{"id":298,"text":"50","start":1645,"end":1647},{"id":299,"text":"000","start":1648,"end":1651},{"id":300,"text":"obyvate\u013emi","start":1652,"end":1662},{"id":301,"text":".","start":1663,"end":1664},{"id":302,"text":"\n\t","start":1578,"end":1580},{"id":303,"text":"\u010eal\u0161\u00edmi","start":1666,"end":1673},{"id":304,"text":"ve\u013ek\u00fdmi","start":1674,"end":1681},{"id":305,"text":"mestami","start":1682,"end":1689},{"id":306,"text":"s\u00fa","start":1690,"end":1692},{"id":307,"text":"Manzini","start":1693,"end":1700},{"id":308,"text":",","start":1701,"end":1702},{"id":309,"text":"Lobamba","start":1703,"end":1710},{"id":310,"text":"a","start":1711,"end":1712},{"id":311,"text":"Siteki","start":1713,"end":1719},{"id":312,"text":".","start":1720,"end":1721},{"id":313,"text":"\n\t","start":1666,"end":1668},{"id":314,"text":"Pri","start":1723,"end":1726},{"id":315,"text":"hraniciach","start":1727,"end":1737},{"id":316,"text":"s","start":1738,"end":1739},{"id":317,"text":"Mozambikom","start":1740,"end":1750},{"id":318,"text":"s\u00fa","start":1751,"end":1753},{"id":319,"text":"pohoria","start":1754,"end":1761},{"id":320,"text":"a","start":1762,"end":1763},{"id":321,"text":"na","start":1764,"end":1766},{"id":322,"text":"severoz\u00e1pade","start":1767,"end":1779},{"id":323,"text":"krajiny","start":1780,"end":1787},{"id":324,"text":"je","start":1788,"end":1790},{"id":325,"text":"da\u017e\u010fov\u00fd","start":1791,"end":1798},{"id":326,"text":"prales","start":1799,"end":1805},{"id":327,"text":".","start":1806,"end":1807},{"id":328,"text":"\n\t","start":1723,"end":1725},{"id":329,"text":"Zvy\u0161ok","start":1809,"end":1815},{"id":330,"text":"krajiny","start":1816,"end":1823},{"id":331,"text":"pokr\u00fdvaj\u00fa","start":1824,"end":1833},{"id":332,"text":"savany","start":1834,"end":1840},{"id":333,"text":".","start":1841,"end":1842},{"id":334,"text":"\n\t","start":1809,"end":1811},{"id":335,"text":"Svazijsko","start":1844,"end":1853},{"id":336,"text":"sa","start":1854,"end":1856},{"id":337,"text":"del\u00ed","start":1857,"end":1861},{"id":338,"text":"na","start":1862,"end":1864},{"id":339,"text":"\u0161tyri","start":1865,"end":1870},{"id":340,"text":"provincie","start":1871,"end":1880},{"id":341,"text":":","start":1881,"end":1882},{"id":342,"text":"V","start":1883,"end":1884},{"id":343,"text":"\u010dele","start":1885,"end":1889},{"id":344,"text":"krajiny","start":1890,"end":1897},{"id":345,"text":"stoj\u00ed","start":1898,"end":1903},{"id":346,"text":"kr\u00e1\u013e","start":1904,"end":1908},{"id":347,"text":".","start":1909,"end":1910},{"id":348,"text":"\n\t","start":1844,"end":1846},{"id":349,"text":"Od","start":1912,"end":1914},{"id":350,"text":"roku","start":1915,"end":1919},{"id":351,"text":"1986","start":1920,"end":1924},{"id":352,"text":"to","start":1925,"end":1927},{"id":353,"text":"je","start":1928,"end":1930},{"id":354,"text":"Mswati","start":1931,"end":1937},{"id":355,"text":"III.","start":1938,"end":1942},{"id":356,"text":",","start":1943,"end":1944},{"id":357,"text":"ktor\u00fd","start":1945,"end":1950},{"id":358,"text":"nahradil","start":1951,"end":1959},{"id":359,"text":"svojho","start":1960,"end":1966},{"id":360,"text":"predchodcu","start":1967,"end":1977},{"id":361,"text":"Sobhuzu","start":1978,"end":1985},{"id":362,"text":"II","start":1986,"end":1988},{"id":363,"text":".","start":1989,"end":1990},{"id":364,"text":"\n\t","start":1912,"end":1914},{"id":365,"text":"Ke\u010f","start":1992,"end":1995},{"id":366,"text":"sa","start":1996,"end":1998},{"id":367,"text":"v","start":1999,"end":2000},{"id":368,"text":"roku","start":2001,"end":2005},{"id":369,"text":"1986","start":2006,"end":2010},{"id":370,"text":"stal","start":2011,"end":2015},{"id":371,"text":"kr\u00e1\u013eom","start":2016,"end":2022},{"id":372,"text":"Mswati","start":2023,"end":2029},{"id":373,"text":"III.","start":2030,"end":2034},{"id":374,"text":",","start":2035,"end":2036},{"id":375,"text":"le\u017eala","start":2037,"end":2043},{"id":376,"text":"pred","start":2044,"end":2048},{"id":377,"text":"n\u00edm","start":2049,"end":2052},{"id":378,"text":"krajina","start":2053,"end":2060},{"id":379,"text":"rozdelen\u00e1","start":2061,"end":2070},{"id":380,"text":"medzi","start":2071,"end":2076},{"id":381,"text":"belo\u0161sk\u00fa","start":2077,"end":2085},{"id":382,"text":"men\u0161inu","start":2086,"end":2093},{"id":383,"text":",","start":2094,"end":2095},{"id":384,"text":"ktor\u00e1","start":2096,"end":2101},{"id":385,"text":"si","start":2102,"end":2104},{"id":386,"text":"chcela","start":2105,"end":2111},{"id":387,"text":"podr\u017ea\u0165","start":2112,"end":2119},{"id":388,"text":"nadvl\u00e1du","start":2120,"end":2128},{"id":389,"text":",","start":2129,"end":2130},{"id":390,"text":"a","start":2131,"end":2132},{"id":391,"text":"\u010derno\u0161sk\u00fa","start":2133,"end":2142},{"id":392,"text":"v\u00e4\u010d\u0161inu","start":2143,"end":2150},{"id":393,"text":",","start":2151,"end":2152},{"id":394,"text":"ktor\u00e1","start":2153,"end":2158},{"id":395,"text":"chcela","start":2159,"end":2165},{"id":396,"text":"roz\u0161\u00edri\u0165","start":2166,"end":2174},{"id":397,"text":"svoje","start":2175,"end":2180},{"id":398,"text":"pr\u00e1va","start":2181,"end":2186},{"id":399,"text":".","start":2187,"end":2188},{"id":400,"text":"\n\t","start":1992,"end":1994},{"id":401,"text":"Bud\u00facnos\u0165","start":2190,"end":2199},{"id":402,"text":"Svazijska","start":2200,"end":2209},{"id":403,"text":"zna\u010dne","start":2210,"end":2216},{"id":404,"text":"ovplyvnia","start":2217,"end":2226},{"id":405,"text":"historick\u00e9","start":2227,"end":2237},{"id":406,"text":"zmeny","start":2238,"end":2243},{"id":407,"text":"odohr\u00e1vaj\u00face","start":2244,"end":2256},{"id":408,"text":"sa","start":2257,"end":2259},{"id":409,"text":"za","start":2260,"end":2262},{"id":410,"text":"hranicami","start":2263,"end":2272},{"id":411,"text":"Juhoafrickej","start":2273,"end":2285},{"id":412,"text":"republiky","start":2286,"end":2295},{"id":413,"text":",","start":2296,"end":2297},{"id":414,"text":"ktorej","start":2298,"end":2304},{"id":415,"text":"\u013eud","start":2305,"end":2308},{"id":416,"text":"si","start":2309,"end":2311},{"id":417,"text":"roku","start":2312,"end":2316},{"id":418,"text":"1994","start":2317,"end":2321},{"id":419,"text":"zvolil","start":2322,"end":2328},{"id":420,"text":"svojho","start":2329,"end":2335},{"id":421,"text":"prv\u00e9ho","start":2336,"end":2342},{"id":422,"text":"\u010derno\u0161sk\u00e9ho","start":2343,"end":2354},{"id":423,"text":"prezidenta","start":2355,"end":2365},{"id":424,"text":"Nelsona","start":2366,"end":2373},{"id":425,"text":"Mandelu","start":2374,"end":2381},{"id":426,"text":".","start":2382,"end":2383},{"id":427,"text":"\n\t","start":2190,"end":2192},{"id":428,"text":"Napriek","start":2385,"end":2392},{"id":429,"text":"tomu","start":2393,"end":2397},{"id":430,"text":"\u017ee","start":2398,"end":2400},{"id":431,"text":"ofici\u00e1lne","start":2401,"end":2410},{"id":432,"text":"existuje","start":2411,"end":2419},{"id":433,"text":"sen\u00e1t","start":2420,"end":2425},{"id":434,"text":"o","start":2426,"end":2427},{"id":435,"text":"po\u010dte","start":2428,"end":2433},{"id":436,"text":"30","start":2434,"end":2436},{"id":437,"text":"\u010dlenov","start":2437,"end":2443},{"id":438,"text":",","start":2444,"end":2445},{"id":439,"text":"Svazijsko","start":2446,"end":2455},{"id":440,"text":"je","start":2456,"end":2458},{"id":441,"text":"absolutistickou","start":2459,"end":2474},{"id":442,"text":"diarchiou","start":2475,"end":2484},{"id":443,"text":",","start":2485,"end":2486},{"id":444,"text":"kde","start":2487,"end":2490},{"id":445,"text":"spolo\u010dn\u00fdmi","start":2491,"end":2501},{"id":446,"text":"hlavami","start":2502,"end":2509},{"id":447,"text":"\u0161t\u00e1tu","start":2510,"end":2515},{"id":448,"text":"s\u00fa","start":2516,"end":2518},{"id":449,"text":"pod\u013ea","start":2519,"end":2524},{"id":450,"text":"trad\u00edcie","start":2525,"end":2533},{"id":451,"text":"kr\u00e1\u013e","start":2534,"end":2538},{"id":452,"text":"a","start":2539,"end":2540},{"id":453,"text":"jeho","start":2541,"end":2545},{"id":454,"text":"matka","start":2546,"end":2551},{"id":455,"text":".","start":2552,"end":2553},{"id":456,"text":"\n\t","start":2385,"end":2387},{"id":457,"text":"Svazijsko","start":2555,"end":2564},{"id":458,"text":"je","start":2565,"end":2567},{"id":459,"text":"rozvojov\u00fd","start":2568,"end":2577},{"id":460,"text":"\u0161t\u00e1t","start":2578,"end":2582},{"id":461,"text":"s","start":2583,"end":2584},{"id":462,"text":"prevahou","start":2585,"end":2593},{"id":463,"text":"po\u013enohospod\u00e1rstva","start":2594,"end":2611},{"id":464,"text":".","start":2612,"end":2613},{"id":465,"text":"\n\t","start":2555,"end":2557},{"id":466,"text":"Z","start":2615,"end":2616},{"id":467,"text":"nerastn\u00fdch","start":2617,"end":2627},{"id":468,"text":"surov\u00edn","start":2628,"end":2635},{"id":469,"text":"sa","start":2636,"end":2638},{"id":470,"text":"\u0165a\u017e\u00ed","start":2639,"end":2643},{"id":471,"text":"\u010dierne","start":2644,"end":2650},{"id":472,"text":"uhlie","start":2651,"end":2656},{"id":473,"text":",","start":2657,"end":2658},{"id":474,"text":"nasleduj\u00fa","start":2659,"end":2668},{"id":475,"text":"diamanty","start":2669,"end":2677},{"id":476,"text":",","start":2678,"end":2679},{"id":477,"text":"\u017eelezn\u00e1","start":2680,"end":2687},{"id":478,"text":"ruda","start":2688,"end":2692},{"id":479,"text":",","start":2693,"end":2694},{"id":480,"text":"zlato","start":2695,"end":2700},{"id":481,"text":",","start":2701,"end":2702},{"id":482,"text":"c\u00edn","start":2703,"end":2706},{"id":483,"text":"a","start":2707,"end":2708},{"id":484,"text":"azbest","start":2709,"end":2715},{"id":485,"text":".","start":2716,"end":2717},{"id":486,"text":"\n\t","start":2615,"end":2617},{"id":487,"text":"V\u00fdznam","start":2719,"end":2725},{"id":488,"text":"m\u00e1","start":2726,"end":2728},{"id":489,"text":"v\u00fdroba","start":2729,"end":2735},{"id":490,"text":"cukru","start":2736,"end":2741},{"id":491,"text":"a","start":2742,"end":2743},{"id":492,"text":"produkcia","start":2744,"end":2753},{"id":493,"text":"dreva","start":2754,"end":2759},{"id":494,"text":".","start":2760,"end":2761},{"id":495,"text":"\n\t","start":2719,"end":2721},{"id":496,"text":"Pestuje","start":2763,"end":2770},{"id":497,"text":"sa","start":2771,"end":2773},{"id":498,"text":"cukrov\u00e1","start":2774,"end":2781},{"id":499,"text":"trstina","start":2782,"end":2789},{"id":500,"text":",","start":2790,"end":2791},{"id":501,"text":"tabak","start":2792,"end":2797},{"id":502,"text":",","start":2798,"end":2799},{"id":503,"text":"kukurica","start":2800,"end":2808},{"id":504,"text":",","start":2809,"end":2810},{"id":505,"text":"ry\u017ea","start":2811,"end":2815},{"id":506,"text":",","start":2816,"end":2817},{"id":507,"text":"bavlna","start":2818,"end":2824},{"id":508,"text":"a","start":2825,"end":2826},{"id":509,"text":"citrusy","start":2827,"end":2834},{"id":510,"text":".","start":2835,"end":2836},{"id":511,"text":"\n\t","start":2763,"end":2765},{"id":512,"text":"Chov\u00e1","start":2838,"end":2843},{"id":513,"text":"sa","start":2844,"end":2846},{"id":514,"text":"hov\u00e4dz\u00ed","start":2847,"end":2854},{"id":515,"text":"dobytok","start":2855,"end":2862},{"id":516,"text":",","start":2863,"end":2864},{"id":517,"text":"kozy","start":2865,"end":2869},{"id":518,"text":"a","start":2870,"end":2871},{"id":519,"text":"hydina","start":2872,"end":2878},{"id":520,"text":".","start":2879,"end":2880},{"id":521,"text":"\n\t","start":2838,"end":2840},{"id":522,"text":"Ve\u013ek\u00e1","start":2882,"end":2887},{"id":523,"text":"\u010das\u0165","start":2888,"end":2892},{"id":524,"text":"Svazij\u010danov","start":2893,"end":2904},{"id":525,"text":"pracuje","start":2905,"end":2912},{"id":526,"text":"v","start":2913,"end":2914},{"id":527,"text":"zlatonosn\u00fdch","start":2915,"end":2927},{"id":528,"text":"baniach","start":2928,"end":2935},{"id":529,"text":"za","start":2936,"end":2938},{"id":530,"text":"hranicami","start":2939,"end":2948},{"id":531,"text":"v","start":2949,"end":2950},{"id":532,"text":"Juhoafrickej","start":2951,"end":2963},{"id":533,"text":"republike","start":2964,"end":2973},{"id":534,"text":".","start":2974,"end":2975}],"meta":{"id":"4009"},"_input_hash":226809449,"_task_hash":-203263869,"_session_id":None,"_view_id":"ner_manual","answer":"accept"}


def visualize_ner(ner_data, out_path='visualized_ner.html'):
    html = displacy.render(ner_data, options={'colors': {'PER': 'yellow', 'MISC': '#f89ff8'}}, manual=True, style="ent", page=True)
    with open(out_path, 'w') as f:
        f.write(html)


def visualize_prodigy(annotations_prodigy):
    annotations_prodigy['ents'] = annotations_prodigy['spans']
    visualize_ner(annotations_prodigy, out_path='visualized_ner_prodigy.html')


def tokenize_nltk(text):
    tokenized = []
    sentences = nltk.sent_tokenize(text, language='czech')
    for sentence in sentences:
        tokenized.append(nltk.tokenize.word_tokenize(sentence, language='czech', preserve_line=True))
    return tokenized


def tokenize_split(text):
    sentences = text.split(' . ')
    return [sentence.split(' ') for sentence in sentences]


def join_tokenized_nltk(tokenized_text):
    return ' '.join([' '.join(sentence) for sentence in tokenized_text])


def connect_annotations(annotations, sentence):
    annotations_s = sorted(annotations, key=lambda x: x['start'])
    connected = []
    tokens = sentence.split(' ')

    annotation_ix = 0
    token_ix = -1
    tokens_len = -1
    tokens_j = ''

    while annotation_ix < len(annotations_s):
        current_annotation = annotations_s[annotation_ix]

        while tokens_len < current_annotation['end']:
            token_ix += 1
            tokens_len += len(tokens[token_ix]) + 1
            tokens_j = tokens_j + tokens[token_ix] + ' '

        if current_annotation['end'] != tokens_len:

            new_annotation = dict(current_annotation)

            while True:
                annotation_ix += 1
                if annotation_ix >= len(annotations_s):
                    break
                current_annotation = annotations_s[annotation_ix]

                new_annotation = {
                    #'text': new_annotation['text'] + current_annotation['text'],
                    'text': sentence[new_annotation['start']:current_annotation['end']],
                    'label': new_annotation['label'],
                    'start': new_annotation['start'],
                    'end': current_annotation['end'],
                }

                if tokens_len <= current_annotation['end']:
                    break

            connected.append(new_annotation)
        else:
            connected.append(current_annotation)

        annotation_ix += 1

    return connected


def filter_annotations(annotations, tokenized_sentence):
    # remove headers like annotations - one word and dot
    if len(tokenized_sentence) == 2 and tokenized_sentence[1] == '.':
        return []

    # remove annotations that dont start with capital letter
    return [ann for ann in annotations if ann['text'][0].isupper()]


def get_clear_annotations(annotations_encoded, text):
    annotations = []
    for annotation in annotations_encoded:
        annotations.append({
            'text': text[annotation['start']:annotation['end']],
            'label': annotation['entity_group'],
            'start': annotation['start'],
            'end': annotation['end'],
        })
    return annotations


def inference(pipeline, tokenized_sentence):
    joined_sentence = ' '.join(tokenized_sentence)
    annotations = pipeline(joined_sentence)
    clear_annotations = get_clear_annotations(annotations, joined_sentence)
    connected_annotations = connect_annotations(clear_annotations, joined_sentence)
    filtered_annotations = filter_annotations(connected_annotations, tokenized_sentence)

    return filtered_annotations


# def inference_wikiann():
#     dataset_path = '../cleaned_data/test_cleaned.txt'
#     n_sentences = 1
#
#     with open(dataset_path) as dataset_file:
#         csv_reader = csv.reader(dataset_file, delimiter=',')
#         sentence_count = 0
#         first_line = True
#         text, tags = [], []
#         for row in csv_reader:
#
#             if first_line == 0:
#                 print(f'Column names are {", ".join(row)}')
#                 first_line = False
#                 continue


def inference_wikidump(dataset_path, pipeline):
    n_documents = 99999
    wikidump_labeled = []

    num_lines = sum(1 for line in open(dataset_path))

    with jsonlines.open(dataset_path) as dataset_file:
        document_count = 0
        print(f'WIKIANN SLOVAK BERT INFERENCING {dataset_path}')
        for doc in tqdm(dataset_file, total=num_lines):

            ner_data = []
            tokenized_text = tokenize_nltk(doc['text'])
            for sentence in tokenized_text:
                annotations = inference(pipeline, sentence)
                ner_data.append({
                    'text': ' '.join(sentence),
                    'ents': annotations,
                })

            document_count += 1
            wikidump_labeled.append({
                'sentences': ner_data,
                'meta': doc['meta'],
            })

            if document_count >= n_documents:
                break
    #visualize_ner({'text': ' '.join(tokenized_text[0]), 'ents': annotations})
    #visualize_ner(ner_data, out_path='visualized_ner_wikidump.html')

    return wikidump_labeled


def find_token_index(span_position, tokens):
    for i, token in enumerate(tokens):
        if token['start'] <= span_position < token['end']:
            return i

    return -1


def wikidump_to_prodigy(wikidump_labeled, output_path):
    docs = []

    print('CONVERTING SLOVAK BERT WIKI ANNOTATIONS TO PRODIGY FORMAT')
    for doc in tqdm(wikidump_labeled):
        tokens = []
        spans = []
        sentences = ''

        for i, sentence in enumerate(doc['sentences']):

            tokenized_sentence = sentence['text'].split(' ')
            tokens_spans = textspan.get_original_spans(tokenized_sentence, sentence['text'])
            tokens.extend([{
                'id': j + len(tokens),
                'text': text,
                'start': spans[0][0] + len(sentences),
                'end': spans[0][1] + len(sentences),
            } for j, (text, spans) in enumerate(zip(tokenized_sentence, tokens_spans))])

            spans.extend([{
                'text': ent['text'],
                'label': ent['label'],
                'start': ent['start'] + len(sentences),
                'end': ent['end'] + len(sentences),
                'token_start': find_token_index(ent['start'] + len(sentences), tokens),
                'token_end': find_token_index(ent['end'] + len(sentences) - 1, tokens),
            } for ent in sentence['ents']])

            sentence_separator = ''
            if i != len(doc['sentences']) - 1:
                sentence_separator = '\n\t'
                tokens.append({
                    'id': len(tokens),
                    'text': sentence_separator,
                    'start': len(sentences),
                    'end': len(sentences) + len(sentence_separator),
                })

            sentences += sentence['text'] + sentence_separator

        doc_prodigy = {
            'text': sentences,
            'spans': spans,
            'tokens': tokens,
            'meta': doc['meta'],
        }
        docs.append(doc_prodigy)

    with jsonlines.open(output_path, mode='w') as writer:
        writer.write_all(docs)


def main():
    ner = pipeline(task='ner', model='./output/test-ner', aggregation_strategy="simple")
    #ner = None

    #dataset_path = '../../wiki_dump/text_jsonl/20211216-171112.jsonl'
    dataset_path = '../data/wikidump/jsonl/20211216-171112.jsonl'

    annotations = inference_wikidump(dataset_path, ner)
    wikidump_to_prodigy(annotations, '../../prodigy/prodigy-data/wikidump/20211216-171112_bert.jsonl')
    #wikidump_to_prodigy(annotations, '../../prodigy/prodigy-data/test03.jsonl')


    #visualize_ner({'text': ' '.join(tokenized_text[0]), 'ents': annotations})


if __name__ == '__main__':
    main()
    #visualize_prodigy(annotations_prodigy)


