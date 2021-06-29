#summary 2 duc 2004
#20/200/110  ---> 33/42/20
#30/200/110  ---> 26/34/09

from summarizer import Summarizer
from summarizer.centroid_embeddings import summarize
from summarizer.mmr_summarizer import MMRsummarize

doc1 = '''Honduras braced for potential catastrophe Tuesday as Hurricane Mitch 
roared through the northwest Caribbean, churning up high waves and 
intense rain that sent coastal residents scurrying for safer ground. 
President Carlos Flores Facusse declared a state of maximum alert 
and the Honduran military sent planes to pluck residents from their 
homes on islands near the coast. At 0900 GMT Tuesday, Mitch was 95 
miles (152 kilometers) north of Honduras, near the Swan Islands. With 
winds near 180 mph (289 kph), and even higher gusts, it was a Category 
5 monster _ the highest, most dangerous rating for a storm. The 350-mile 
(560-kilometer) wide hurricane was moving west at 8 mph (12 kph). 
``Mitch is closing in,'' said Monterrey Cardenas, mayor of Utila, 
an island 20 miles (32 kilometers) off the Honduran coast. ``And God 
help us.'' Mitch posed no immediate threat to the United States, forecasters 
said, but was expected to remain in the northwest Caribbean for five 
days. The U.S. National Weather Service in Miami said Mitch could 
weaken somewhat, but warned it would still remain ``a very dangerous 
hurricane capable of causing catastrophic damage.'' The entire coast 
of Honduras was under a hurricane warning and up to 15 inches (38 
centimeters) of rain was forecast in mountain areas. The Honduran 
president closed schools and public offices on the coast Monday and 
ordered all air force planes and helicopters to evacuate people from 
the Islas de la Bahia, a string of small islands off the country's 
central coast. The head of the Honduran armed forces, Gen. Mario Hung 
Pacheco, said 5,000 soldiers were standing by to help victims of the 
storm, but he warned the military could not reach everyone. ``For 
that humanitarian work, we would need more than 300 Hercules C-137 
planes,'' he said. ``Honduras doesn't have them.'' A hurricane warning 
was also in effect for the Caribbean coast of Guatemala. In Belize, 
a hurricane watch was in place and the government also closed schools 
and sent workers home early Monday. Panic buying stripped bread from 
the shelves of some stores and some gasoline stations ran dry. Coastal 
Belize City was hit so hard by Hurricane Hattie in 1961 that the country 
built a new capital inland at Belmopan. Mexico mobilized troops and 
emergency workers Monday on the east coast of the Yucatan peninsula, 
which was also under a hurricane watch, and Cuba said it had evacuated 
600 vacationers from the Island of Youth. Jerry Jarrell, the weather 
center director, said Mitch was the strongest hurricane to strike 
the Caribbean since 1988, when Gilbert killed more than 300 people. 
In La Ceiba, on Honduras' northern coast, people stood in long lines 
at filling stations Monday to buy gasoline under a steady rain. Maria 
Gonzalez said she needed the gas to cook with when her firewood gets 
wet. Still, she bought only 37 cents worth _ all she could afford. 
``I have six children, and we live in a riverbed,'' she said. ``If 
it gets really bad, we'll go to the church and see what the architect 
of the world has in store for us.'' Swinwick Jackson, a fisherman 
on Utila, had tied up his boats and was taking his family to stay 
with a relative on higher ground. National police spokesman Ivan Mejia 
said the Coco, Segovia and Cruta rivers all overflowed their banks 
Monday along Honduras' eastern coast. ``Frightened people are moving 
into the mountains to search for shelter,'' he said. In El Progreso, 
100 miles (160 kilometers) north of the Honduran capital of Tegucigalpa, 
the army evacuated more than 5,000 people who live in low-lying banana 
plantations along the Ulua River, said Nolly Soliman, a resident. 
Before bearing down on Honduras, Mitch swept past Jamaica and the 
Cayman Islands. Rain squalls flooded streets in the Jamaican capital, 
Kingston, and government offices and schools closed in the Caymans, 
a British colony of 28,000 people. The strongest hurricane to hit 
Honduras in recent memory was Fifi in 1974, which ravaged Honduras' 
Caribbean coast, killing at least 2,000 people. '''

doc2='''Hurricane Mitch paused in its whirl through the western Caribbean 
on Wednesday to punish Honduras with 120-mph (205-kph) winds, topping 
trees, sweeping away bridges, flooding neighborhoods and killing at 
least 32 people. Mitch was drifting west at only 2 mph (3 kph) over 
the Bay Islands, Honduras' most popular tourist area. It also was 
only 30 miles (50 kms) off the coast, and hurricane-force winds stretched 
outward 105 miles (165 kms); tropical storm-force winds 175 miles 
(280 kms). That meant the Honduran coast had been under hurricane 
conditions for more than a day. ``The hurricane has destroyed almost 
everything,'' said Mike Brown, a resident of Guanaja Island which 
was within miles (kms) of the eye of the hurricane. ``Few houses have 
remained standing.'' At its, 4th graf pvs '''

doc3='''Hurricane Mitch cut through the Honduran coast like a ripsaw Thursday, 
its devastating winds whirling for a third day through resort islands 
and mainland communities. At least 32 people were killed and widespread 
flooding prompted more than 150,000 to seek higher ground. Mitch, 
once among the century's most powerful hurricanes, weakened today 
as it blasted this Central American nation, bringing downpours that 
flooded at least 50 rivers. It also kicked up huge waves that pounded 
seaside communities. The storm's power was easing and by 1200 GMT, 
it had sustained winds of 80 mph (130 kph), down from 100 mph (160 
kph) around midnight and well below its 180 mph (290 kph) peak of 
early Tuesday. After remaining virtually stationary for more than 
a day, the U.S. National Hurricane Center said Thursday the center 
of the 350-mile-wide (560-kilometer-wide) storm had moved slightly 
to the south but remained just off the Honduran coast. Hurricane-force 
winds whirled up to 30 miles (50 kilometers) from the center, with 
rain-laden tropical storm winds extending well beyond that. Caught 
near the heart of the storm were the Bay Islands, about 25 miles (40 
kilometers) off Honduras' coast and popular with divers and beachcombers. 
``The hurricane has destroyed almost everything,'' said Mike Brown, 
a resident of Guanaja Island, 20 miles (32 kilometers) off the coast. 
``Few houses have remained standing.'' Honduran officials said 14 
people had died on that small island alone, and at least nine had 
died elsewhere in the country. More than 72,000 people had been evacuated 
to shelters. Nine other deaths had been reported elsewhere in the 
region by early Thursday _ more than a day after Mitch drifted to 
just off the coast and seemed to park there. An American was thrown 
from his boat south of Cancun, Mexico, on Monday and was presumed 
dead. Eight others died in Nicaragua in flooding. Honduran officials 
said more than 200 towns and villages had been isolated by the storm, 
left without power, telephones or clean drinking water. Agriculture 
Minister Pedro Arturo Sevilla said crucial grain, citrus and banana 
crops had been damaged ``and the economic future of Honduras is uncertain.'' 
Rain-swollen rivers knocked out bridges and roads, isolating La Ceiba, 
a coastal city of 40,000 people located 80 miles (128 kilometers) 
from the Bay Islands. About 10,000 residents fled to crowded shelters 
in schools, churches and firehouses. While supplies of food and gasoline 
seemed to hold up, drivers worried about the coming days formed long 
lines to fill their tanks at gas stations and some supermarkets took 
measures to limit panic buying. La Ceiba officials appealed for pure 
water for those in shelters and some residents set out plastic buckets 
to collect rainwater. Only a few hotels and offices with their own 
generators had electricity. Wind-whipped waves almost buried some 
houses near the shore. People evacuated low-lying houses by wading 
through chest-deep water with sodden bags of belongings on their heads. 
In neighboring Belize, most of the 75,000 residents of coastal Belize 
City had left by Wednesday, turning the country's largest city into 
a ghost town. Police and soldiers patrolled the streets, and a few 
people wandered amid the boarded-up houses. The cable television company 
was broadcasting only The Weather Channel. With the storm seemingly 
anchored off Honduras, officials in Mexico to the north eased emergency 
measures on the Caribbean coast of the Yucatan Peninsula, where hundreds 
of people remained in shelters as a precaution Wednesday night. More 
than 20,000 tourists had abandoned Cancun and nearby resort areas, 
leaving hotels at about 20 percent of capacity. Houston accountant 
Kathy Montgomery said that she and her friend Nina Devries had tried 
to leave Cancun but found all the flights full. ``It's been horrible,'' 
said Montgomery, as she and her friend drank cocktails at an outdoor 
restaurant. ``We couldn't go out on a boat, we couldn't go snorkeling. 
``Even Carlos' N Charlie's and Senor Frog's are closed,'' she said 
dejectedly, referring to two restaurants. ``Some vacation.'' The U.S. 
Agency for International Development sent two helicopters each to 
Belize and Honduras to help in search, rescue and relief efforts. 
At its peak, Mitch was the fourth-strongest Caribbean hurricane in 
this century, behind Gilbert in 1988, Allen in 1980 and the Labor 
Day hurricane of 1935. '''

doc4='''At least 231 people have been confirmed dead in Honduras from former-hurricane 
Mitch, bringing the storm's death toll in the region to 357, the National 
Emergency Commission said Saturday. Mitch _ once, 2nd graf pvs'''

doc5='''In Honduras, at least 231 deaths have been blamed on Mitch, the National 
Emergency Commission said Saturday. El Salvador _ where 140 people 
died in flash floods _ declared a state of emergency Saturday, as 
did Guatemala, where 21 people died when floods swept away their homes. 
Mexico reported one death from Mitch last Monday. In the Caribbean, 
the U.S. Coast Guard widened a search for a tourist schooner with 
31 people aboard that hasn't been heard from since Tuesday. By late 
Sunday, Mitch's winds, once near 180 mph (290 kph), had dropped to 
near 30 mph (50 kph), and the storm _ now classified as a tropical 
depression _ was near Tapachula, on Mexico's southern Pacific coast 
near the Guatemalan border. Mitch was moving west at 8 mph (13 kph) 
and was dissipating but threatened to strengthen again if it moved 
back out to sea. '''

doc6='''Nicaraguan Vice President Enrique Bolanos said Sunday night that between 
1,000 and 1,500 people were buried in a 32-square mile (82.88 square-kilometer) 
area below the slopes of the Casita volcano in northern Nicaragua. 
That is in addition to least another 600 people elsewhere in the country, 
Bolanos said.'''

doc7='''BRUSSELS, Belgium (AP) - The European Union on Tuesday approved 6.4 
million European currency units (dlrs 7.7 million) in aid for thousands 
of victims of the devastation caused by Hurricane Mitch in Central 
America. EU spokesman Pietro Petrucci said the funds will be used 
to provide basic care such as medicine, food, water sanitation and 
blankets to thousands of people whose homes were destroyed by torrential 
rains and mudslides. The aid will be distributed in Nicaragua, El 
Salvador, Honduras and Guatemala which have most suffered from Mitch's 
deadly passage, the EU executive Commission said in a statement. Officials 
in Central America estimated Tuesday that about 7,000 people have 
died in the region. The greatest losses were reported in Honduras, 
where an estimated 5,000 people died and 600,000 people _ 10 percent 
of the population _ were forced to flee their homes after last week's 
storm. El Salvador's National Emergency Committee listed 174 dead, 
96 missing and 27,000 homeless. But its own regional affiliate in 
San Miguel province reported 125 dead there alone. Guatemala reported 
100 storm-related deaths. The latest EU aid follows an initial 400,000 
ecu (dlrs 480,000). the EU approved for the region on Friday. The 
full 6.8 million ecu (dlrs 8.18 million) will be channeled through 
humanitarian groups working in the region.'''

doc8='''Pope John Paul II appealed for aid Wednesday for the Central American 
countries stricken by hurricane Mitch and said he feels close to the 
thousands who are suffering. Speaking during his general audience, 
the pope urged ``all public and private institutions and all men of 
good will'' to do all they can ``in this grave moment of destruction 
and death.'' Hurricane Mitch killed an estimated 9,000 people throughout 
Central America in a disaster of such proportions that relief agencies 
have been overwhelmed. Among those attending the audience were six 
Russian cosmonauts taking a special course in Italy. As a gift, they 
gave John Paul a spacesuit. '''

doc9='''Better information from Honduras' ravaged countryside enabled officials 
to lower the confirmed death toll from Hurricane Mitch from 7,000 
to about 6,100 on Thursday, but leaders insisted the need for help 
was growing. President Carlos Flores declared Hurricane Mitch had 
set back Honduras' development by 50 years. He urged the more than 
1.5 million Hondurans affected by the storm to help with the recovery 
effort. ``The county is semi-destroyed and awaits the maximum effort 
and most fervent and constant work of every one of its children,'' 
he said. In the capital, Tegucigalpa, Mexican rescue teams began searching 
for avalanche victims. Honduran doctors dispensed vaccinations to 
prevent disease outbreaks in shelters crammed with refugees. As of 
Thursday, Mitch had killed 6,076 people in Honduras _ down from officials' 
earlier estimate of 7,000. The numbers of missing dropped from an 
estimated 11,000 to 4,621, Government Minister Delmer Urbizo said. 
``We have more access to places affected by the storm,'' Urbizo explained. 
``Until now, we have had a short amount of time and few resources 
to get reliable information.'' In Nicaragua, around 2,000 people were 
killed, most of those swept away when a volcano crater lake collapsed 
a week ago. El Salvador reported 239 dead; Guatemala said 194 of its 
people had been killed. Six people died in southern Mexico and seven 
in Costa Rica. Aid groups and governments have called for other countries 
to send medicine, water, canned food, roofing materials and equipment 
to help deliver supplies. In Washington on Thursday, President Bill 
Clinton ordered dlrs 30 million in Defense Department equipment and 
services and dlrs 36 million in food, fuel and other aid be sent to 
Honduras, Nicaragua, El Salvador and Guatemala. The White House also 
said Clinton was dispatching Tipper Gore, wife of Vice President Al 
Gore, to Central America on a mission to show the U.S. commitment 
to providing humanitarian relief. Hillary Rodham Clinton also will 
travel to the region, visiting Nicaragua and Honduras on Nov. 16. 
She later will stop in El Salvador and Guatemala before continuing 
on to Haiti and the Dominican Republic for a visit that had been canceled 
due to Hurricane Georges, which struck the Caribbean in October. Mitch, 
which sat off the Honduran coast for several days last week, destroyed 
scores of Central American communities before moving northwest. The 
weakened storm crossed southern Florida on Thursday, damaging mobile 
homes and buildings and injuring at least seven people. Countries 
overwhelmed by the storm's devastation have only just begun to calculate 
the damage. Honduran authorities still don't know how many shelters 
have been set up across the Central American country. Surveyors have 
yet to evaluate roughly 10 percent of the most affected areas _ the 
departments of Cortes, Atlantida, Colon and Yoro in the north, southern 
Choluteca and Valle, and the central department of Francisco Morazan, 
which includes the capital. Numbers still can vary wildly. The estimated 
number of homeless dropped from 580,000 to 569,000 Thursday. Mitch 
damaged or destroyed at least 90 bridges on major highways, including 
most spans on Highway 5, the main north-south route, officials said. 
Urbizo said Honduran officials hoped the scale of need in next-door 
Nicaragua wouldn't overshadow Honduras' plight. ``Our problem is that 
all of the country has been affected,'' he said. Thursday's relief 
operations paled in comparison to the scope of the disaster. A total 
of 14 helicopter relief missions were delivering aid to stricken towns, 
said Col. Roger Antonio Caceres of the government's Operation Mitch 
emergency response task force. ``We can manage with the number of 
aircraft we have because there is little to distribute,'' Caceres 
said. More help was on the way. Mexico was sending 700 tons of food, 
11 tons of medicine, at least 12 helicopters, four cargo planes and 
475 soldiers to help in relief operations. The United States committed 
19 Blackhawk and Chinook helicopters, two C-27 aircraft and one C-130 
cargo plane. France said it was sending 250 rescue workers to Central 
America, along with a ship loaded with construction material and equipment. 
The U.N. World Food Program said Thursday it was diverting ships, 
some already at sea, to rush their cargoes of donated food to Central 
America. It also was pulling food from warehouses at its base in Rome, 
probably for delivery by emergency relief flights, spokesman Trevor 
Rowe said. ``We're trying to move food as fast as possible to help 
people as soon as possible,'' Rowe said. Searches for the missing 
continued, and some decomposed bodies, once found, were being buried 
in common graves. About 100 victims had been buried around Tegucigalpa, 
Mayor Nahum Valladeres said. In the flood-ravaged Tegucigalpa neighborhood 
of Nueva Esperanza, Mexican military rescuers loaded search dogs on 
their backs and forded a muddy river to look for people believed buried 
in a 200-foot (60-meter) avalanche that occurred last Friday. Dozens 
of homes were swept into the river. ``This is the first place we've 
been'' with the dogs, Honduran army Maj. Freddy Diaz Celaya said. 
``From here we'll continue searching downriver.'' Concerned that crowded 
shelter conditions could produce outbreaks of hepatitis, respiratory 
infections and other ailments, the Health Ministry announced an inoculation 
campaign, especially for children. Doctors volunteering at a shelter 
housing 4,000 people at Tegucigalpa's Polytechnic Development Institute 
said they'd heard of the campaign but had yet to receive word or medicines 
from the Health Ministry. ``We have to vaccinate the children,'' said 
Dr. Mario Soto, who has treated at least 300 children at the shelter 
for diarrhea, conjunctivitis and bacterial infections. '''

doc10='''Aid workers struggled Friday to reach survivors of Hurricane Mitch, 
who are in danger of dying from starvation and disease in the wake 
of the storm that officials estimate killed more than 10,000 people. 
Foreign aid and pledges of assistance poured into Central America, 
but damage to roads and bridges reduced the amount of supplies reaching 
hundreds of isolated communities to a trickle: only as much as could 
be dropped from a helicopter, when the aircraft can get through. In 
the Aguan River Valley in northern Honduras, floodwaters have receded, 
leaving a carpet of mud over hundreds of acres (hectares). In many 
nearby villages, residents have gone days without potable water or 
food. A 7-month-old baby died in the village of Olvido after three 
days without food. Residents feared more children would die. ``The 
worst thing, the saddest thing, are the children. The children are 
suffering, even dying,'' said the Rev. Cecilio Escobar Gallindo, the 
parish priest. A score of cargo aircraft landed Thursday at the normally 
quiet Toncontin airport in the Honduran capital of Tegucigalpa, delivering 
aid from Mexico, the United States, Japan and Argentina. Former U.S. 
President Jimmy Carter and his wife, Rosalynn, intended to visit Nicaragua 
on Friday to learn more about the hurricane's impact, The Carter Center 
in Atlanta announced. ``We hope this visit will help call attention 
to the suffering and humanitarian need this disaster has created,'' 
Carter said in a statement. U.S. President Bill Clinton requested 
a ``global relief effort'' to help Central America and boosted U.S. 
emergency aid to dlrs 70 million. Clinton is dispatching a delegation 
next week led by Tipper Gore, wife of Vice President Al Gore, to deliver 
some of the supplies destined for Honduras, Nicaragua, El Salvador 
and Guatemala. First lady Hillary Rodham Clinton added Nicaragua and 
Honduras to a trip she plans to the region beginning Nov. 16. Taiwan 
said today it will donate dlrs 2.6 million in relief to Honduras, 
Nicaragua, El Salvador and Guatemala. The four countries are among 
a dwindling number of nations that recognize Taiwan, which China claims 
is a breakaway province. Two British ships that were in the area on 
an exercise were on their way to Honduras to join relief efforts, 
the Defense Ministry said Friday. ``It's a coincidence that the ships 
are there but they've got men and equipment that can be put to work 
in an organized way,'' said International Development Secretary Clare 
Short. Nicaragua said Friday it will accept Cuba's offer to send doctors 
as long as the communist nation flies them in on its own helicopters 
and with their own supplies. Nicaraguan leaders previously had refused 
Cuba's offer of medical help, saying it did not have the means to 
transport or support the doctors. Nicaragua's leftist Sandinistas, 
who maintained close relations with Fidel Castro during their 1979-90 
rule, had criticized the refusal by President Arnoldo Aleman's administration.'''

body=doc1+doc2+doc3+doc4+doc5+doc6+doc7+doc8+doc9+doc10



#summarize(doc2)
model = Summarizer()
###embedding+kmeans result
print('\n\n1st stage\n\n')
result = model(body, min_length=60)
full = ''.join(result)

f = open('projects/test-summarization/system/task1_englishSyssum1.txt','w')
print (full,file=f)
f.close()





#perform centroid word embedding
print('\n\n2nd stage\n\n')
result2=summarize(full)
#print('\n\n2nd stage\n\n')
#print(result2)
f = open('projects/test-summarization/system/task1_englishSyssum2.txt','w')
print (result2,file=f)
f.close()





print('\n\n3rd stage\n\n')
result3=MMRsummarize(result2)
#print(result3)
f = open('projects/test-summarization/system/task1_englishSyssum3.txt','w')
print (result3,file=f)
f.close()




