#ifndef CONV3_PARAMS_H
#define CONV3_PARAMS_H

#include "constants.h"

// input 12x12x10, ch_num = 12

const float conv3_filter_weight[CONV3_K_NUM * NUM_10CH *  FILTER_SIZE * FILTER_SIZE] ={
	0.05457707494497299, -0.0661604031920433, 0.09924643486738205,
	0.08744901418685913, 0.05072982236742973, 0.061657559126615524,
	0.027808090671896935, -0.09960898756980896, 0.0424095056951046,
	-0.04372796416282654, 0.11196833848953247, 0.03757844865322113,
	-0.030042752623558044, 0.1342461109161377, 0.048532698303461075,
	0.08819947391748428, -0.08930779248476028, 0.0035991142503917217,
	0.023146722465753555, 0.07331497967243195, -0.010884412564337254,
	0.12739403545856476, 0.063301682472229, 0.06317053735256195,
	0.0559891015291214, 0.08946696668863297, -0.08276168256998062,
	-0.024359408766031265, 0.08573039621114731, 0.019188150763511658,
	0.12350010126829147, -0.039556387811899185, 0.06684254854917526,
	0.04360499233007431, -0.09781592339277267, 0.09881591796875,
	0.06541663408279419, -0.005766017362475395, -0.0586407296359539,
	-0.09665720164775848, -0.014229107648134232, -0.08646178245544434,
	-0.0544344037771225, 0.021685950458049774, 0.10293018072843552,
	-0.11066194623708725, -0.002192761516198516, 0.024358874186873436,
	0.006226441357284784, 0.06036845222115517, 0.05717000737786293,
	0.09662525355815887, -0.013211417011916637, -0.027610521763563156,
	0.021699681878089905, 0.08104416728019714, 0.009662487544119358,
	0.05368756875395775, -0.010481076315045357, -0.11228980869054794,
	0.0681525468826294, 0.11154226958751678, 0.036195117980241776,
	-0.015223231166601181, 0.06274709105491638, -0.028886791318655014,
	0.13108621537685394, 0.1381317377090454, 0.06882346421480179,
	0.05181731656193733, -0.06575561314821243, 0.06598470360040665,
	0.04579385370016098, 0.23470240831375122, 0.1633690595626831,
	0.2627975046634674, 0.32370221614837646, 0.24473902583122253,
	-0.056871406733989716, -0.03684660419821739, -0.06709524989128113,
	-0.005161541514098644, -0.0901203528046608, -0.017825786024332047,
	-0.03233210742473602, -0.0067017520777881145, 0.13157910108566284,
	0.02344164252281189, 0.0830719992518425, 0.07140274345874786,
	0.10128030925989151, 0.0919991135597229, 0.07271161675453186,
	0.08183418959379196, 0.07812464982271194, -0.011349650099873543,
	0.023785851895809174, -0.09793718159198761, -0.08701951801776886,
	-0.01233238261193037, 0.09504066407680511, -0.025713885203003883,
	0.08102815598249435, 0.07925287634134293, -0.061136748641729355,
	-0.0016326544573530555, -0.07612134516239166, 0.04495696350932121,
	0.09314072132110596, -0.022658588364720345, 0.025647617876529694,
	-0.014703062362968922, -0.047776639461517334, -0.0315132662653923,
	-0.037567075341939926, 0.07215016335248947, -0.018951674923300743,
	0.07818296551704407, 0.02396911382675171, -0.06080637499690056,
	0.038699012249708176, -0.04932883009314537, 0.029722874984145164,
	-0.09493031352758408, 0.014708038419485092, 0.06479878723621368,
	0.07642575353384018, -0.014261962845921516, -0.047054603695869446,
	-0.08187910169363022, -0.0889180600643158, -0.04859335720539093,
	-0.00698377238586545, -0.015365909785032272, -0.10379400849342346,
	-0.06909455358982086, 0.016344724223017693, -0.00606282614171505,
	0.07475949823856354, -0.009636679664254189, 0.09372121095657349,
	0.015378734096884727, 0.10563136637210846, -0.03793375566601753,
	-0.03701405227184296, 0.08461228013038635, 0.07302884757518768,
	-0.04466650262475014, 0.033675726503133774, -0.06548623740673065,
	0.08165416866540909, 0.046234723180532455, 0.01237480528652668,
	0.07867419719696045, -0.008247402496635914, -0.08274654299020767,
	0.07067840546369553, -0.05620138347148895, -0.06052016839385033,
	0.0006656062323600054, -0.020190097391605377, -0.037181269377470016,
	0.16556450724601746, 0.2220420390367508, 0.1237790584564209,
	0.024124452844262123, 0.005257143639028072, 0.012632397934794426,
	-0.10444636642932892, -0.04795485734939575, 0.10549362748861313,
	0.06113288551568985, -0.03084307163953781, 0.06664008647203445,
	-0.07753698527812958, -0.03569592162966728, -0.03373255208134651,
	-0.017050782218575478, 0.09270912408828735, -0.017251737415790558,
	-0.10309804975986481, -0.01994919776916504, 0.08117261528968811,
	-0.023471524938941002, 0.07195574045181274, 0.08196395635604858,
	-0.00403305608779192, -0.09368100762367249, 0.008220991119742393,
	0.02816505916416645, 0.1815129816532135, -0.0375736728310585,
	0.16792334616184235, -0.03356104716658592, -0.04548342153429985,
	0.010376574471592903, -0.04773501679301262, -0.1306600570678711,
	0.0654028058052063, 0.16937348246574402, -0.07036858797073364,
	0.19585072994232178, 0.005744233727455139, 0.056284256279468536,
	-0.013018857687711716, -0.11356697231531143, -0.09218278527259827,
	0.1254456490278244, 0.02017829567193985, 0.020629141479730606,
	-0.03645816445350647, 0.09719253331422806, 0.010907853953540325,
	-0.06470806896686554, -0.10558540374040604, -0.05748077109456062,
	-0.06600203365087509, 0.027409086003899574, 0.0004635895602405071,
	0.03367122262716293, 0.09024132788181305, -0.033125072717666626,
	0.0948476567864418, -0.08482975512742996, -0.02140895277261734,
	0.03110082447528839, -0.07333999127149582, 0.005957371089607477,
	0.07641314715147018, 0.05115111172199249, 0.06365318596363068,
	-0.059527426958084106, 0.13538946211338043, 0.01575431227684021,
	0.027698861435055733, -0.036013975739479065, 0.03525608777999878,
	0.02572559379041195, -0.11301332712173462, -0.0612214021384716,
	0.11783579736948013, 0.08005091547966003, -0.07935530692338943,
	0.053037501871585846, 0.16030186414718628, -0.05685078725218773,
	0.09051843732595444, 0.035285674035549164, 0.028235306963324547,
	0.11730396747589111, -0.10732904821634293, -0.06137334182858467,
	0.3318130671977997, 0.4744723439216614, 0.14631162583827972,
	0.46001917123794556, 0.3862442076206207, 0.28426027297973633,
	0.0852273479104042, -0.14009732007980347, -0.06504429876804352,
	0.057623934000730515, 0.05951567366719246, 0.1122828871011734,
	-0.0462198369204998, 0.189236119389534, 0.11427225172519684,
	0.06366143375635147, 0.017986014485359192, -0.0036022525746375322,
	-0.034473128616809845, -0.05367710068821907, 0.08637969195842743,
	-0.08470293134450912, 0.06662128865718842, 0.05657373368740082,
	0.03808826953172684, 0.022316372022032738, -0.007025120779871941,
	0.02027510106563568, -0.08191761374473572, -0.026176035404205322,
	0.03927312046289444, 0.05977439880371094, -0.07732237130403519,
	0.03017333894968033, -0.03296298161149025, 0.07704809308052063,
	0.026949740946292877, -0.12393099069595337, 0.04338213428854942,
	-0.10704059153795242, 0.047015994787216187, -0.03831984102725983,
	0.11001849174499512, 0.0808463841676712, 0.07900285720825195,
	0.04243572801351547, 0.07317882031202316, -0.06206964701414108,
	0.07646383345127106, -0.0770472064614296, 0.08812712132930756,
	0.03508641570806503, 0.00016290585335809737, 0.00770330335944891,
	0.026898443698883057, 0.02247794158756733, -0.07290861755609512,
	0.04411712661385536, 0.10155946016311646, -0.005909701809287071,
	0.03203091770410538, 0.015247168019413948, 0.004677894990891218,
	-0.06550246477127075, 0.021665163338184357, -0.057023633271455765,
	-0.07643239200115204, 0.005654593929648399, 0.07214438170194626,
	-0.11090674996376038, -0.06543981283903122, 0.09694437682628632,
	-0.03848881646990776, 0.03493982180953026, -0.08490090817213058,
	-0.07335461676120758, 0.014914768747985363, 0.06928566843271255,
	0.06282493472099304, 0.09919476509094238, 0.08436886966228485,
	-0.07740692794322968, -0.13066810369491577, 0.0720098614692688,
	-0.07417259365320206, -0.03541869297623634, 0.002978402888402343,
	0.06435248255729675, 0.11412352323532104, -0.046932850033044815,
	-0.016060665249824524, -0.008414474315941334, -0.030954184010624886,
	0.0540451817214489, -0.022449975833296776, 0.07133176177740097,
	0.00030050953500904143, -0.0326024666428566, 0.018786247819662094,
	0.00605480233207345, -0.08617020398378372, 0.022488122805953026,
	0.09440117329359055, -0.014006377197802067, -0.10287357121706009,
	0.0447000190615654, 0.0013158763758838177, 0.0995735302567482,
	0.0042430986650288105, 0.09234191477298737, 0.017087677493691444,
	-0.09393628686666489, 0.0525786466896534, -0.10645707696676254,
	0.009271329268813133, -0.10552529990673065, -0.05831153318285942,
	-0.00846972968429327, -0.001229597139172256, 0.0565350167453289,
	-0.01588250696659088, -0.10192219167947769, 0.039088454097509384,
	-0.054406993091106415, -0.05414359271526337, -0.11989754438400269,
	0.07071356475353241, 0.08764059096574783, 0.07667023688554764,
	0.1168384850025177, 0.008481882512569427, -0.02794617787003517,
	-0.0012959068408235908, -0.047507427632808685, -0.029174521565437317,
	-0.09571269899606705, -0.05741539224982262, -0.08113370090723038,
	0.013875235803425312, 0.06355255842208862, 0.059215154498815536,
	-0.08101235330104828, 0.02824416384100914, -0.08153587579727173,
	-0.09446088969707489, -0.09733747690916061, -0.0979708880186081,
	-0.0881698802113533, -0.02466423809528351, -0.058950889855623245,
	0.06373140960931778, 0.05673898011445999, 0.02465193159878254,
	0.036491718143224716, -0.05722995102405548, 0.009017987176775932,
	-0.02129187248647213, -0.0011151350336149335, 0.10024826973676682,
	-0.010517138987779617, 0.11437958478927612, 0.07553036510944366,
	0.020835518836975098, 0.05567723512649536, -0.029227377846837044,
	0.027406683191657066, 0.05742776021361351, -0.06883646547794342,
	0.017536884173750877, 0.10005149245262146, -0.01583520509302616,
	0.12609100341796875, 0.11858808249235153, 0.13801540434360504,
	-0.07037711888551712, 0.008695561438798904, 0.10957889258861542,
	-0.011864137835800648, -0.09098107367753983, -0.09916415065526962,
	0.013596557080745697, 0.2502126097679138, 0.2161756157875061,
	-0.013426270335912704, 0.1900104284286499, 0.2325146496295929,
	-0.25015026330947876, -0.2233685702085495, -0.12087561190128326,
	-0.0872342512011528, 0.0374908484518528, -0.038747627288103104,
	-0.02428749017417431, -0.04099101200699806, -0.029684701934456825,
	-0.11883507668972015, 0.03278638795018196, -0.07663078606128693,
	-0.05317093804478645, 0.010737312957644463, 0.08428676426410675,
	-0.08028177171945572, -0.05036373808979988, 0.010236270725727081,
	-0.04494621977210045, -0.048171427100896835, 0.08445579558610916,
	0.030890973284840584, -0.0422184132039547, 0.03875347971916199,
	-0.06228933483362198, 0.0919976755976677, -0.011048068292438984,
	-0.08496125042438507, -0.049677710980176926, -0.10181469470262527,
	0.060246583074331284, 0.07511866092681885, 0.05854751169681549,
	0.05310807004570961, -0.09258072078227997, 0.02544788457453251,
	0.0673878863453865, 0.060231443494558334, 0.08368580788373947,
	-0.047665439546108246, -0.08888587355613708, -0.013815654441714287,
	0.02133890613913536, -0.0375061109662056, -0.015597761608660221,
	0.06348176300525665, 0.020392319187521935, -0.08398017287254333,
	-0.06718311458826065, 0.01624659076333046, -0.0802544429898262,
	-0.06231486797332764, -0.006634248420596123, -0.04378702864050865,
	-0.10384251177310944, 0.09007932990789413, -0.027213891968131065,
	-0.012750733643770218, -0.06895611435174942, -0.10089776664972305,
	-0.08686292916536331, 0.015932995826005936, -0.03815396875143051,
	0.031153110787272453, -0.028865711763501167, -0.016802167519927025,
	-0.10116191953420639, 0.08941581845283508, -0.05957149341702461,
	-0.09096319228410721, 0.011740017682313919, 0.07846499979496002,
	-0.010875086300075054, 0.049636729061603546, -0.08982549607753754,
	0.03225353732705116, -0.07846330106258392, 0.10557921230792999,
	-0.038795050233602524, 0.008935962803661823, 0.03666790947318077,
	-0.03540106862783432, 0.041148360818624496, -0.0070114233531057835,
	-0.06659305840730667, -0.0019454432185739279, 0.07804732769727707,
	-0.020675022155046463, -0.0015934404218569398, -0.009751216508448124,
	-0.08758316934108734, -0.08444152027368546, -0.012857997789978981,
	-0.0651434063911438, -0.051363054662942886, -0.05386728420853615,
	-0.05410708859562874, -0.04727989435195923, -0.04416230693459511,
	0.07209920883178711, -0.08752023428678513, 0.035418737679719925,
	-0.006398503668606281, 0.003782982937991619, 0.04896831512451172,
	0.013993535190820694, 0.007068851962685585, 0.051763489842414856,
	0.05079963430762291, 0.06881850212812424, -0.08108232170343399,
	-0.04358958080410957, 0.03244369477033615, 0.11051177233457565,
	0.018416281789541245, -0.1003534197807312, 0.07523667067289352,
	-0.010811855085194111, -0.07383644580841064, -0.04801195114850998,
	0.005186490248888731, 0.11806084960699081, 0.022295556962490082,
	-0.08143816888332367, -0.03204735368490219, -0.006707562133669853,
	-0.07951197773218155, 0.09097569435834885, 0.08126872032880783,
	0.07092930376529694, 0.12797823548316956, 0.009897122159600258,
	0.09931311756372452, 0.05476343631744385, 0.09916364401578903,
	0.06979294866323471, -0.013940396718680859, 0.04328448697924614,
	0.022015104070305824, 0.004223356489092112, 0.10420970618724823,
	0.0242497306317091, -0.07526838034391403, 0.03851809725165367,
	0.07490089535713196, -0.050107188522815704, -0.03815672919154167,
	-0.03366763889789581, 0.05693316459655762, -0.021300379186868668,
	0.06305786222219467, 0.043286412954330444, -0.06950310617685318,
	0.04618821293115616, -0.06883206218481064, -0.03882009908556938,
	0.06782907247543335, 0.012496626935899258, 0.0997319146990776,
	-0.08183705806732178, 0.08183743059635162, -0.08589695394039154,
	0.0029905240517109632, 0.14408618211746216, -0.0814741924405098,
	-0.04339488968253136, 0.11003798246383667, 0.12795530259609222,
	-0.08306895196437836, 0.11539988219738007, 0.05728088691830635,
	0.04487387835979462, 0.12354490160942078, 0.08777154237031937,
	0.06360116600990295, 0.17985422909259796, 0.31116870045661926,
	-0.22935038805007935, -0.004893000237643719, 0.42992621660232544,
	-0.23245659470558167, 0.02391841821372509, 0.22578033804893494,
	0.07990507781505585, 0.07703597098588943, 0.06911613792181015,
	-0.1043725311756134, 0.011215273290872574, 0.10790304094552994,
	-0.09851433336734772, -0.08036795258522034, 0.09426455944776535,
	-0.014084497466683388, 0.03204473853111267, -0.04482937976717949,
	0.0655178651213646, 0.0388251468539238, 0.08046483993530273,
	-0.07857587933540344, 0.01373486127704382, -0.10502196848392487,
	-0.038517724722623825, -0.1061444953083992, 0.13642637431621552,
	0.08309759199619293, 0.013479172252118587, 0.08997579663991928,
	0.10159018635749817, 0.05438956245779991, 0.011207396164536476,
	-0.03532930091023445, -0.01762031577527523, 0.11211364716291428,
	0.08256204426288605, 0.10101179033517838, 0.16442649066448212,
	-0.05583205446600914, 0.13861046731472015, 0.16377824544906616,
	0.010887081734836102, -0.017750045284628868, -0.04984663426876068,
	-0.04068439081311226, 0.09625685214996338, -0.03179387375712395,
	0.05569585785269737, 0.1517896056175232, -0.058600518852472305,
	-0.0446782149374485, -0.036820705980062485, -0.04101019725203514,
	-0.03143475577235222, -0.08472353965044022, 0.013958812691271305,
	0.09763776510953903, 0.022628840059041977, -0.09250883758068085,
	0.08701513707637787, 0.011323241516947746, 0.008291330188512802,
	-0.0027048480696976185, 0.013163558207452297, -0.042928703129291534,
	0.062191154807806015, 0.028490865603089333, 0.03386246785521507,
	0.0981839969754219, -0.012004834599792957, -0.08092772960662842,
	-0.002081109443679452, 0.007790735457092524, 0.09627161175012589,
	0.0790017768740654, 0.1307549774646759, 0.06660641729831696,
	-0.10050662606954575, -0.08122453093528748, -0.04561448469758034,
	0.06630044430494308, 0.016311530023813248, 0.19736462831497192,
	0.03806658461689949, 0.11705807596445084, 0.007652292028069496,
	0.009259238839149475, -0.07955390959978104, 0.15658283233642578,
	-0.03359859064221382, 0.2861109972000122, 0.4911119043827057,
	-0.07505122572183609, 0.26328638195991516, 0.31350114941596985,
	0.11397676914930344, -0.04045470803976059, 0.0005843836115673184,
	0.04160383716225624, -0.07725641131401062, 0.0038773079868406057,
	-0.052374400198459625, 0.09087935090065002, 0.10741300135850906,
	0.03948960825800896, -0.02524920180439949, 0.08139323443174362,
	-0.002324749017134309, 0.032223206013441086, -0.05816130340099335,
	0.007852952927350998, -0.040216006338596344, 0.06473448127508163,
	0.013301868923008442, -0.08784519135951996, -0.10032812505960464,
	-0.01289293821901083, -0.10995025187730789, 0.006695680785924196,
	0.023089243099093437, 0.003886200487613678, 0.003743794048205018,
	-0.03578998148441315, -0.0187502633780241, 0.0027048783376812935,
	-0.09236430376768112, -0.043818239122629166, 0.03444783017039299,
	0.07582936435937881, 0.08283183723688126, -0.0062704505398869514,
	-0.026454715058207512, -0.09111469238996506, 0.1096869483590126,
	0.04749864712357521, 0.0711076483130455, 0.025952374562621117,
	0.026179227977991104, 0.059255536645650864, 0.02380509115755558,
	-0.0006323381094262004, -0.09394451230764389, -0.09616518020629883,
	0.04777965694665909, -0.05366828292608261, 0.034160416573286057,
	0.04569443687796593, 0.08051556348800659, -0.08267039805650711,
	0.022764122113585472, -0.04252873361110687, -0.05003475025296211,
	0.022995837032794952, -0.0831073671579361, 0.06768757849931717,
	-0.0024380586110055447, -0.058042462915182114, 0.014480003155767918,
	0.09448807686567307, -0.02354133315384388, 0.06715211272239685,
	-0.02377573773264885, 0.060456693172454834, 0.06466235965490341,
	0.08675059676170349, -0.09814520925283432, -0.037896253168582916,
	-0.07587811350822449, 0.06293758749961853, 0.0717182382941246,
	-0.0922398641705513, 0.09011983871459961, -0.04413331300020218,
	0.08594487607479095, 0.025747962296009064, -0.01580870896577835,
	-0.027187181636691093, -0.023287082090973854, 0.08544967323541641,
	-0.023763395845890045, -0.11285574734210968, 0.01620119996368885,
	0.047273389995098114, -0.023429328575730324, 0.0870361253619194,
	-0.09759028255939484, 0.054193660616874695, 0.08605398237705231,
	-0.03711888566613197, -0.03843718767166138, 0.021832171827554703,
	-0.11171076446771622, 0.015243331901729107, -0.021236665546894073,
	-0.08259019255638123, 0.07298681139945984, 0.09379833191633224,
	0.015421733260154724, -0.09509267657995224, -0.03842529281973839,
	0.07230691611766815, 0.09201467037200928, -0.07875646650791168,
	-0.028984207659959793, 0.0008848520228639245, -0.012280882336199284,
	0.11077218502759933, 0.19904018938541412, 0.08388444036245346,
	0.12728358805179596, 0.019471775740385056, -0.11383186280727386,
	0.06156608462333679, 0.11451776325702667, 0.00193190504796803,
	0.12076137959957123, 0.2428916096687317, 0.09321177005767822,
	0.03611889109015465, 0.0639868900179863, 0.1346854418516159,
	0.05186530575156212, -0.03558943048119545, -0.0638127475976944,
	0.10423344373703003, 0.15479792654514313, 0.09466960281133652,
	0.07377889007329941, 0.0646517276763916, 0.03993956372141838,
	-0.03488717973232269, 0.07864197343587875, -0.07070174068212509,
	0.10661456733942032, 0.09893516451120377, 0.08814781904220581,
	-0.03142101690173149, 0.041093844920396805, -0.008665288798511028,
	0.041127704083919525, 0.027923723682761192, 0.08194655179977417,
	-0.042964253574609756, 0.0387752540409565, -0.002897415542975068,
	-0.034394726157188416, -0.09227297455072403, 0.041008081287145615,
	0.07166554033756256, 0.026900891214609146, -0.01533627137541771,
	0.11829441040754318, 0.08282613754272461, 0.05411165580153465,
	-0.027811570093035698, -0.0723634883761406, 0.03533566743135452,
	-0.07113602757453918, 0.07923489809036255, -0.01865367777645588,
	0.12400246411561966, 0.15303172171115875, 0.011996863409876823,
	0.08627969771623611, 0.15490466356277466, 0.03803243860602379,
	0.01811697706580162, 0.101800836622715, 0.19928035140037537,
	0.26236680150032043, 0.5054551959037781, 0.4655786454677582,
	0.2892360985279083, 0.238102525472641, 0.22344882786273956,
	0.07913296669721603, 0.03241448104381561, 0.03372793644666672,
	-0.07933545112609863, -0.023102376610040665, 0.2229226976633072,
	-0.02626761980354786, 0.1883789598941803, 0.08615556359291077,
	0.04844200983643532, -0.09945874661207199, 0.05357169359922409,
	0.048523686826229095, -0.088555246591568, 0.09363959729671478,
	-0.058727800846099854, -0.013547993265092373, 0.07426156103610992,
	0.004288745112717152, -0.015370747074484825, -0.025131387636065483,
	-0.004184975288808346, 0.006662323605269194, -0.11762592196464539,
	0.01509256474673748, -0.05280115827918053, -0.012307416647672653,
	0.10283791273832321, -0.00047085501137189567, 0.022489219903945923,
	-0.032718148082494736, -0.007617362774908543, -0.12075576931238174,
	0.12643316388130188, 0.09118237346410751, -0.02181771770119667,
	-0.05359470844268799, 0.053233228623867035, -0.08527781814336777,
	0.07391684502363205, -0.0303594209253788, -0.028133807703852654,
	0.09285742044448853, 0.028672516345977783, 0.06194369122385979,
	-0.10456966608762741, -0.04492973908782005, 0.08990364521741867,
	-0.015941031277179718, -0.07196864485740662, -0.09479012340307236,
	0.05059950053691864, 0.05173184722661972, -0.06599244475364685,
	-0.0037192138843238354, 0.08829902112483978, -0.03521313518285751,
	-0.07976914942264557, -0.019353995099663734, 0.028223751112818718,
	0.008117861114442348, 0.01592741720378399, 0.09484460204839706,
	-0.017326446250081062, -0.02958369255065918, -0.0869317576289177,
	-0.04101794213056564, -0.05022542178630829, 0.013003678061068058,
	0.08661965280771255, 0.061160072684288025, 0.03781689703464508,
	0.0014843237586319447, 0.061330296099185944, -0.06239679083228111,
	0.13302786648273468, -0.02200467512011528, -0.061760857701301575,
	-0.01858585886657238, -0.07905314117670059, 0.06676534563302994,
	-0.051658060401678085, 0.07622209191322327, 0.012268432416021824,
	0.08426805585622787, 0.13829614222049713, -0.16844835877418518,
	0.11125558614730835, 0.09227042645215988, -0.1585189700126648,
	0.08280801773071289, 0.02122538350522518, 0.11272136121988297,
	0.1069389283657074, 0.07228296995162964, -0.04720652848482132,
	-0.005215306766331196, -0.04115840047597885, 0.06983792781829834,
	-0.09352970868349075, 0.04596675932407379, 0.047914449125528336,
	-0.06268518418073654, -0.019778866320848465, -0.02210310846567154,
	-0.09519664943218231, -0.0985337644815445, -0.10322554409503937,
	-0.005966649856418371, -0.06938837468624115, -0.0018611146369948983,
	0.055103059858083725, -0.030789965763688087, 0.061989475041627884,
	-0.09587696939706802, 0.0002768925332929939, 0.08197062462568283,
	-0.030879957601428032, -0.03844895586371422, -0.10151982307434082,
	0.04853902757167816, 0.05012898147106171, 0.008646665140986443,
	0.05536028742790222, 0.07221154123544693, 0.04490623250603676,
	-0.061455175280570984, 0.033388201147317886, -0.0659308210015297,
	0.06545049697160721, -0.05328384414315224, 0.039069682359695435,
	0.007666293065994978, 0.03909468278288841, -0.08809006959199905,
	0.002084850799292326, 0.06931637227535248, 0.03682122007012367,
	-0.07111956924200058, 0.003931274637579918, -0.04435903951525688,
	-0.025711115449666977, 0.0865657851099968, 0.03159378841519356,
	-0.053308773785829544, -0.008289485238492489, -0.07780498266220093,
	-0.05700232833623886, 0.013300903141498566, -0.030229590833187103,
	0.03363717347383499, -0.05939609929919243, -0.10088635981082916,
	-0.011827479116618633, -0.05091913416981697, 0.043250225484371185,
	0.077357716858387, 0.05639044940471649, -0.08243675529956818,
	0.011910575442016125, -0.05181884765625, -0.02157294563949108,
	-0.04312818869948387, -0.10523150116205215, 0.04542221501469612,
	-0.051272641867399216, -0.07798980921506882, -0.09392086416482925,
	-0.07432971149682999, 0.04402507096529007, 0.006150584667921066,
	0.10241616517305374, -0.07608712464570999, -0.07056882977485657,
	0.08609426766633987, -0.11414404213428497, 0.10689374059438705,
	-0.11058201640844345, -0.0965774804353714, 0.10800658166408539,
	0.03780234605073929, -0.06605113297700882, -0.031436722725629807,
	0.027750160545110703, 0.05315191298723221, -0.06913865357637405,
	0.07377086579799652, -0.06217849254608154, 0.057857856154441833
};


const float conv3_filter_bias[CONV3_K_NUM] = {
	-0.030910756438970566, -0.028695354238152504, 0.11311472952365875, -0.08908182382583618, 0.061449598520994186, 
	-0.08668774366378784, 0.052980825304985046, 0.05048935487866402, -0.044937316328287125, -0.024724487215280533, 
	0.08546501398086548, -0.06789018958806992};

#endif // CONV3_PARAMS_H