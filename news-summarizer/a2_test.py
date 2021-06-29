
#30/200/110 ---> 28 / 32/ 5
#30/150/110 ---> 25 / 29/ 5
#20/200/110 ---> 26 / 30/ 5
#40/200/110 ---> 25 / 34/ 6


from summarizer import Summarizer
from summarizer.centroid_embeddings import summarize
from summarizer.mmr_summarizer import MMRsummarize

doc1 = '''Exxon Corp. and Mobil Corp. have held discussions about combining 
their business operations, a person involved in the talks said Wednesday. 
It was unclear Wednesday whether talks were continuing. If the companies 
were to merge, it would create the largest U.S. company in terms of 
revenue. A possible merger was reported separately by both The Financial 
Times of London and Bloomberg News. The reported talks between Exxon, 
whose annual revenue exceeds that of Wal-Mart and General Electric, 
and Mobil, the No. 2 U.S. oil company, come as oil prices have sunk 
to their lowest in almost 12 years. A combined company would be bigger 
than Royal Dutch/Shell Group, the world's largest oil company by revenue. 
Financial terms of the discussions could not be determined Wednesday. 
Neither Exxon or Mobil would comment. Any union would reunite two 
parts of John D. Rockefeller's Standard Oil Trust, which was broken 
up by the Supreme Court in 1911. Exxon was then known as Standard 
Oil of New Jersey, and Mobil consisted of two companies: Standard 
Oil of New York and Vacuum Oil. Both Exxon, which has a market value 
of $176.7 billion, and Mobil, which has a market value of $61.1 billion, 
have a history of being fiercely independence. Both have already cut 
back on staff and made themselves lean in order to survive long periods 
when oil prices are low. But this has been a particularly unsettling 
year for the oil industry, and there is little prospect that crude 
oil prices will recover soon. Consequently, chief executives of most 
oil companies have had to swallow their pride and look for suitable 
partners. This summer, British Petroleum announced a $48.2 billion 
agreement to buy Amoco Corp., creating the world's third-largest oil 
company and prompting analysts to predict even more widespread consolidation. 
``It showed that megamergers are doable,'' said Adam Sieminski, an 
analyst for BT Alex. Brown. He added, however, that a combination 
between Exxon and Mobil would not be an easy match because Mobil has 
been known for being a proud company that has said in the past that 
it would not want to merge. Exxon, he added, is a ``well run company 
that likes to grow its own businesses.'' He added that the heads of 
both companies, Lee Raymond, the chairman of Exxon, which is based 
in Irving, Texas, and Lucio Noto, the chairman, president and chief 
executive of Mobil, which is based in Fairfax, Va., are different 
personalities. ``It will not be easy,'' he said of combining the two 
far-flung companies, which have vast networks of refineries and gas 
stations that overlap in the United States and Europe. ``If you offer 
enough money you can make anything happen,'' he added. Both companies 
are under pressure to find new fields of oil to help them survive 
in the long term. Like other oil companies, they had hoped to quickly 
tap into the vast reserves of Russia. Even though they were prepared 
to spend billions, they have held back because of the political and 
economic crisis in Russia and great reluctance by Russian officials 
and oil companies to give up control of vast fields. Thus they have 
had to fall back on their own exploration areas such as the deep waters 
in the Gulf of Mexico and West Africa. Such exploration is very expensive, 
and even when large fields are found it often takes platforms costing 
$1 billion to bring the oil into production. Oil prices have been 
under pressure for more than a year, falling more than 40 percent 
from the $20-a-barrel level because of growing inventories of petroleum 
and declining Asian demand caused by the economic crisis there. On 
Wednesday, crude oil for January delivery fell 29 cents, or 2.6 percent, 
to $11.86 a barrel on the New York Mercantile Exchange, close to the 
12-year low of $11.42 reached on June 15. Members of the Organization 
of Petroleum Exporting Countries and some other oil-producing nations, 
notably Mexico, have tried to stem the price drops with pledges to 
cut back on production. But those pledges have not always been honored, 
and rallies in the oil market this year have proven short-lived. OPEC 
members on Wednesday continued their discussion on extending their 
production cutbacks, and an agreement is expected as early as Thursday. 
In the spring, OPEC agreed to reduce production by 2.6 million barrels 
a day, about 3 percent of the daily world supply of 74 million barrels. 
The main result of that agreement appears to have been to keep oil 
prices from falling below $10 a barrel. Washington regulators said 
Wednesday that they had not been notified about the Exxon-Mobil discussions. 
The Federal Trade Commission is still reviewing British Petroleum's 
pending purchase of Amoco. An Exxon-Mobil deal would be certain to 
receive several months' worth of scrutiny by the commission, which 
would review how much of the industry such a merger would control. 
Analysts and investment bankers were split about the logic of the 
possible merger. Some pointed to difficulties that the companies could 
face if they were combined. ``If you asked me if Exxon needed to be 
bigger, the answer is probably no,'' said Garfield Miller, president 
of Aegis Energy Advisors Corp., a small independent investment bank 
based in New York. ``It is hard to say that there is anything in particular 
to gain.'' In particular, Miller said, the two companies have enormous 
similarities in their domestic refining and marketing businesses. 
``They really do overlap quite a bit,'' he said. ``You really do wonder 
what is the benefit of all that redundancy.'' Another investment banker 
in the energy business, speaking on the condition of anonymity, also 
questioned the rationale for the discussed merger. ``When you look 
at the BP-Amoco deal, you can rationalize it,'' the banker said. ``But 
none of those reasons apply to an Exxon-Mobil deal.'' But Amy Jaffe, 
an energy research analyst with the James A. Baker III Institute for 
Public Policy, said the combination of the two companies would be 
logical, in part because it would give them greater influence in bidding 
for projects in Middle Eastern countries. ``This is a deal that makes 
sense,'' Ms. Jaffe said. ``With this combined company, there is no 
project that would be too big.'' In addition, Ms. Jaffe said, the 
merger would provide each company with new oil and gas assets in areas 
of the world where they had little influence. ``There are a lot of 
complimentary assets where they are not redundant,'' she said. She 
said that Exxon, for example, has a strong presence in Angola, while 
Mobil does not. And Mobil has significant assets in the Caspian Sea 
and Nigeria, where Exxon is weak. '''

doc2='''Exxon Corp. and Mobil Corp. have held discussions about combining 
their business operations, a person involved in the talks said Wednesday. 
It was unclear Wednesday whether talks were continuing. If the companies 
were to merge, it would create the largest U.S. company in terms of 
revenue. A possible merger was reported separately by both The Financial 
Times of London and Bloomberg News. The reported talks between Exxon, 
whose annual revenue exceeds that of General Electric Co., and Mobil, 
the No. 2 U.S. oil company, came as oil prices sank to their lowest 
in almost 12 years. A combined company would be bigger than Royal 
Dutch/Shell Group, the world's largest oil company by revenue. Financial 
terms of the discussions could not be determined Wednesday. Neither 
Exxon or Mobil would comment. Any union would reunite two parts of 
John D. Rockefeller's Standard Oil Trust, which was broken up by the 
Supreme Court in 1911. Exxon was then known as Standard Oil of New 
Jersey, and Mobil consisted of two companies: Standard Oil of New 
York and Vacuum Oil. As oil prices have plummeted to levels last seen 
in the mid-1980s, oil companies have been under pressure to cut costs. 
Exxon, which has a market value of $176.7 billion, and Mobil, which 
has a market value of $61.1 billion, both have histories of being 
fiercely independent, and both have already cut back on staff and 
made themselves lean to survive even during a prolonged period of 
low oil prices. But this has been a particularly unsettling year for 
the oil industry, and there is little prospect that crude oil prices 
will recover soon. Consequently, chief executives of most oil companies 
have had to swallow their pride and look for suitable partners. This 
summer, British Petroleum announced an agreement to buy Amoco Corp. 
for $48.2 million, creating the world's third-largest oil company 
and prompting analysts to predict even more widespread consolidation. 
``It showed that megamergers are doable,'' said Adam Sieminski, an 
analyst for BT Alex. Brown. He added, however, that any combination 
between Exxon and Mobil would not be an easy match because Mobil has 
been known for being a proud company that has said in the past that 
it would not want to merge. Exxon, he added, is a ``well-run company 
that likes to grow its own businesses.'' He added that the heads of 
both companies, Lee Raymond, the chairman of Exxon, which is based 
in Irving, Texas, and Lucio Noto, the chairman, president and chief 
executive of Mobil, which is based in Fairfax, Va., are different 
personalities. ``It will not be easy,'' he said of combining the two 
far-flung companies, which have vast networks of refineries and gas 
stations that overlap in the United States and Europe. ``If you offer 
enough money you can make anything happen,'' he added. Both companies 
are under pressure to find new fields of oil to help them survive 
in the long term. Like other oil companies, they had hoped to quickly 
tap into the vast reserves of Russia. Even though they were prepared 
to spend billions, they have held back because of the political and 
economic crisis in Russia and great reluctance by Russian officials 
and oil companies to give up control of vast fields. Thus they have 
had to fall back on exploration areas of their own such as the deep 
waters in the Gulf of Mexico as well as West Africa and parts of Asia. 
Such exploration is very expensive, and even when a big field is discovered, 
platforms costing $1 billion or more are required to bring the it 
into production. Oil prices have been under pressure for more than 
a year, falling more than 40 percent from the $20-a-barrel level because 
of growing inventories of petroleum and declining Asian demand caused 
by the economic crisis there. On Wednesday, crude oil for January 
delivery fell 29 cents, or 2.6 percent, to $11.86 a barrel on the 
New York Mercantile Exchange, close to the 12-year low of $11.42 reached 
on June 15. Members of the Organization of Petroleum Exporting Countries 
and some other oil-producing nations, notably Mexico, have tried to 
stem the price drops with pledges to cut back on production. But those 
pledges have not always been honored, and rallies in the oil market 
this year have proven short-lived. OPEC members on Wednesday continued 
their discussion on extending their production cutbacks, and an agreement 
is expected as early as Thursday. In the spring, OPEC agreed to reduce 
production by 2.6 million barrels a day, about 3 percent of the daily 
world supply of 74 million barrels. The main result of that agreement 
appears to have been to keep oil prices from falling below $10 a barrel. 
Washington regulators said Wednesday that they had not been notified 
about the Exxon-Mobil discussions. The Federal Trade Commission is 
still reviewing British Petroleum's pending purchase of Amoco. An 
Exxon-Mobil deal would be certain to receive several months' worth 
of scrutiny by the commission, which would review how much of the 
industry the combined company would control. Analysts and investment 
bankers were split about the logic of a potential deal. Some pointed 
to difficulties that the companies could face if they were combined. 
``If you asked me if Exxon needed to be bigger, the answer is probably 
no,'' said Garfield Miller, president of Aegis Energy Advisors Corp., 
a small independent investment bank based in New York. ``It is hard 
to say that there is anything in particular to gain.'' In particular, 
Miller said, the two companies have enormous similarities in their 
domestic refining and marketing businesses. ``They really do overlap 
quite a bit,'' he said. ``You really do wonder what is the benefit 
of all that redundancy.'' Another investment banker in the energy 
business, speaking on the condition of anonymity, also questioned 
the rationale for the discussed merger. ``When you look at the BP-Amoco 
deal, you can rationalize it,'' the banker said. ``But none of those 
reasons apply to an Exxon-Mobil deal.'' But Amy Jaffe, an energy research 
analyst with the James A. Baker III Institute for Public Policy, said 
the combination of the two companies would be logical, in part because 
it would give them greater influence in bidding for development projects 
in the Middle East. ``This is a deal that makes sense,'' Ms. Jaffe 
said. ``With this combined company, there is no project that would 
be too big.'' Ms. Jaffe said the proposed deal would provide each 
company with assets in areas where it had little influence. ``There 
are a lot of complementary assets where they are not redundant,'' 
she said. She said that Exxon, for example, has a strong presence 
in Angola, while Mobil does not. And Mobil has significant assets 
in the Caspian Sea and Nigeria, where Exxon is weak.'''

doc3='''Whether or not the talks between Exxon and Mobil lead to a merger 
or some other business combination, America's economic history is 
already being rewritten. In energy as in businesses like financial 
services, telecommunications and automobiles, global competition and 
technology have made unthinkable combinations practical, even necessary. 
Oil companies like Exxon Corp. and Mobil Corp. have an additional 
pressure, one unthinkable less than two decades ago. Crude oil prices 
have fallen sharply, plunging 40 percent just this year to levels, 
adjusted for inflation, not seen since before the first oil embargo 
25 years ago. As such, the oil companies, having spent years cutting 
their costs, are desperate for further savings in order to continue 
operating profitably with such low prices. Exxon and Mobil are the 
two largest, strongest competitors to emerge from the nation's most 
famous antitrust case, the 1911 breakup of John D. Rockefeller's Standard 
Oil Trust. Now they face a Royal Dutch/Shell Group that is larger 
than either of them. They will also confront a British Petroleum PLC 
made far more potent in the United States by its agreement this summer 
to buy Amoco Corp. for $48.2 billion. Industry executives say further 
deals on this scale are inevitable. Executives at both companies did 
not return calls Thursday for comment on the talks, and it was unclear 
Thursday night what the outcome might be. But if Exxon and Mobil agree 
to become one, antitrust regulators are likely to be cautious about 
putting back together much of what they long ago broke apart. Even 
so, most oil industry analysts contend that improved efficiency from 
combining giant energy companies would do more to lower costs than 
the more concentrated ownership of gas stations and refineries would 
do to raise them. ``The ultimate beneficiary of all this will be the 
consumer,'' said Daniel Yergin, the chairman of Cambridge Energy Research 
Associates. If Exxon and Mobil ultimately do combine, the costs could 
prove heaviest for energy industry employees. Analysts say that of 
the about 80,000 global employees at Exxon, based in Irving, Texas, 
and the more than 40,000 at Mobil, in Fairfax, Va., thousands would 
be likely to lose their jobs. Exxon, with Lee R. Raymond, and Mobil, 
with Lucio A. Noto, both have chief executives who have been preoccupied 
with the humbling accommodations that low oil prices have made necessary. 
Oil companies were everybody's favorite targets during the trust-busting 
era early this century and again during the Arab oil embargoes of 
the 1970s. Now they seem especially vulnerable as demand weakens in 
much of the world, especially in economically troubled Asia, weighing 
further on already depressed prices. ``They're pitiful, helpless giants,'' 
said Ronald Chernow, the author of ``Titan,'' a biography of John 
D. Rockefeller, Standard Oil's founder. As such, these giants are 
compelled to continue cutting costs and spreading the risks of their 
huge, expensive international projects that are needed to develop 
oil reserves needed for the next century. Mobil, with $58.4 billion 
in sales last year, might seem large enough to undertake anything. 
But in competing for rights to develop huge natural gas fields in 
Turkmenistan, a former Soviet republic, Mobil was unable to match 
Shell's offer to build a pipeline for $1 billion or more. Oil companies 
have decided that they cannot count on a rebound in oil prices to 
revive their fortunes any time soon. Earlier this month, the Energy 
Department predicted that the collapse of Asian demand would continue 
to depress oil prices for nearly a decade, and by as much as $5.50 
a barrel in the year 2000. And Thursday, the Organization of Petroleum 
Exporting Countries put off until March any decision on extending 
their oil production cutbacks to prop up prices. Moreover, improving 
technology for exploration and production and the opening of new regions 
to development have added to the already huge supply of oil that is 
on hand now. In response, many energy companies have already begun 
a new wave of cutbacks in their staffs and operations. To further 
reduce costs, companies like Mobil are forming partnerships that stop 
short of full mergers. Two years ago, Mobil agreed to combine its 
European refining and marketing operations with British Petroleum's, 
resulting in annual savings of about $500 million. Shell and Texaco 
then formed a refining partnership in the United States. In the face 
of these partnerships, said Amy Jaffe, an energy analysts with the 
James A. Baker III Institute for Public Policy in Houston, ``if you're 
an Exxon, how do you compete?'' Though Mobil and Exxon might have 
high concentrations of gas stations in certain areas of the United 
States, analysts say they have far more competition at the pump than 
before oil prices collapsed in the 1980s. Thousands of convenience 
stores now also sell gasoline produced by a variety of refining companies, 
and foreign national oil companies, like Venezuela's, sell their supplies 
through acquired companies like Citgo. Under Noto, Mobil has been 
hunting for ways of becoming large and lean enough to survive. He 
took the lead in the deal with British Petroleum and has considered 
buying up smaller companies. And he has made it clear that corporate 
or personal pride would never block a deal. In the European agreement 
with British Petroleum, Mobil's red flying horses have come down from 
the fronts of many gas stations, while the green and yellow BP logos 
have gone up. As major oil companies team up, said John Hervey, an 
analyst with Donaldson Lufkin &AMP; Jenrette, ``If the price is right, 
egos will not get in the way.'' In the past, Mobil reportedly had 
further talks with British Petroleum about other combinations, as 
well as talks with Amoco about forming an American refining venture. 
The company was also interested in Conoco Inc. when DuPont Co. began 
looking to divest itself of that oil and gas business. But no deals 
were reached. More recently, there was speculation on Wall Street 
that Mobil had been talking with Chevron Corp. British Petroleum purchase 
of Amoco put more pressure on Noto to seek a deal, Hervey said. ``I 
don't think this whole thing would have started if British Petroleum 
had not pulled the trigger,'' he said. The British Petroleum-Amoco 
deal, progressing quickly since the August announcement towards an 
expected $2 billion in annual savings, has put even companies as large 
as Exxon on the spot. There, Raymond has so far concentrated more 
on making his operations more efficient than on finding allies. Though 
Royal Dutch/Shell's financial resources are greater than Exxon's, 
some analysts say Exxon still has the size and soundness to absorb 
a company of Mobil's size. Only the largest oil companies can afford 
to take advantage of today's best opportunities. Finding new fields 
in the deep waters off the coast of West Africa or in the Gulf of 
Mexico can require platforms costing $1 billion or more. Producing 
oil in the states of the former Soviet Union has taken more time and 
money than many investors anticipated. In September, Noto was among 
the American executives invited by a Saudi leader visiting Washington 
to greatly step up investments in his country, too. Noto spent from 
1977 to 1985 in Saudi Arabia himself, building up Mobil operations 
including a huge refinery. But any new partnerships with the Saudis 
might require both Mobil's connections and Exxon's capital. Back when 
Standard Oil organized its operations by state, Exxon was Standard 
Oil of New Jersey, while Mobil was Standard Oil of New York. Even 
after the Standard Oil monopoly was broken up, they remained for several 
years in the same building in Manhattan. Chernow, the Rockefeller 
biographer, noted that two pieces of the Standard Oil trust are already 
likely to be united. British Petroleum bought Standard Oil of Ohio 
in the 1970s, and Amoco was once Standard Oil of Indiana. The break-up 
of Standard Oil and the resulting competition has often been cited 
as a precedent for the current antitrust action against Microsoft. 
Chernow sees no reason why allowing an Exxon acquisition of Mobil 
to go through should suggest more leniency for Microsoft. ``Exxon 
would not be obviously larger than its leading competitors, the way 
Microsoft is,'' he said. In the energy business today, he added, ``there 
are other large dinosaurs that stalk that particular jungle.``'''

doc4='''News that Exxon and Mobil, two giants in the energy patch, were in 
merger talks last week is the biggest sign yet that corporate marriages 
are back in vogue. Even before that combination came to light, deal-making 
was fast and furious. On Monday alone, $40.4 billion in corporate 
acquisitions were either announced or declared imminent. Driving the 
resurgence in mergers is a roaring stock market, the recognition by 
major corporations that it is getting harder to increase revenues 
internally and growing confidence among market players that the economy 
will not plunge into a recession next year. There are also industry-specific 
issues, like low crude-oil prices that are driving oil giants into 
one another's arms. But for investors, mega-marriages are not where 
the real money is to be made. Rather, it is among smaller companies, 
whose still-depressed stock prices are luring bigger acquirers with 
stocks that again are near their peaks. If Exxon buys Mobil at close 
to current prices, deals this month will have a total value of more 
than $140 billion _ off from April's peak of $244 billion but three 
times the volume in September, when the stock market was falling. 
Which industries are likely to witness the most mergers? Tom Burnett, 
director of Merger Insight, an institutional investment advisory firm 
in New York, says more deals are a certainty in energy, which is suffering 
from low crude-oil prices. Burnett also says health care executives 
are finding it tougher than ever to lift earnings. But smaller companies 
may be a better way to play the takeover game. Charles LaLoggia, editor 
of the Special Situation Investor newsletter in Potomac, Md., said: 
``Some of the premiums in high-profile mergers aren't so great anymore. 
The values are in small-cap stocks.'' Another reason is that deals 
involving smaller-cap candidates are less likely to incur the wrath 
of antitrust regulators. LaLoggia reckons that the odds of picking 
takeover winners increase if an investor focuses on companies already 
partially owned by another. In the energy sector, Houston Exploration 
qualifies, he says; the oil and gas driller is 66 percent owned by 
Keyspan Energy. He also believes more deals are imminent among drug 
chains and supermarkets. Longs Drug Stores and Drug Emporium, he says, 
remain acquisition candidates, though neither is controlled by another 
concern. In supermarkets, LaLoggia likes the Great Atlantic &AMP; 
Pacific Tea Co., 55 percent owned by Tengelmann Group of Germany. 
Another pick is Smart and Final, an operator of warehouse-style stores 
that is 55 percent owned by the U.S. subsidiary of Groupe Casino, 
France's largest supermarket chain. ``Both A&AMP;P and Smart and Final 
are trading at just about book value,'' LaLoggia said. Takeovers typically 
occur above book value. Finally, he recommends National Presto Industries, 
a maker of housewares and electrical appliances, which is trading 
near $39, close to its low for the year. The company has $30 a share 
in cash on its balance sheet, no debt and a dividend yield of 5 percent. 
After last month's announced acquisition of Rubbermaid by the Newell 
Co., LaLoggia thinks National Presto could find itself in an acquirer's 
cross-hairs. '''

doc5='''Times are tough in the oil patch. Still, it boggles the mind to accept 
the notion that hardship is driving profitable Big Oil to either merge, 
as British Petroleum and Amoco have already agreed to do, or at least 
to consider the prospect, as Exxon and Mobil are doing. Still, Big 
Oil and small oil are getting squeezed by low petroleum prices and 
the high capital costs of exploration. Given the exotic locales of 
the most promising, untapped fields, it seems unlikely that exploration 
will get cheaper. And with West Texas crude trading at around $12 
a barrel, it seems a safe bet oil that won't be selling for $100 a 
barrel by the turn of the century _ a price some analysts in the early 
1980s were predicting it would reach. Philip K. Verleger Jr., publisher 
of Petroleum Economics Monthly and a senior adviser to the Brattle 
Group, a Cambridge, Mass., consulting firm, spent some time late last 
week talking about Mobil, Exxon and the changing dynamics of the oil 
business. Following are excerpts from the conversation: Q. (italics)There 
is a lot of focus on the antitrust aspects of an Exxon-Mobil deal. 
Do you see any problems?(end italics) A. Let me say right off that 
I don't think this is a done deal. I think it is far from that. But 
if it were to happen, I don't see many problems. BP Amoco is the perfect 
end-to-end merger, one in which there is little or no overlap with 
the company you are merging with. Exxon-Mobil comes close. The first 
issue is competition in local markets. The only possible problem area 
there is on the West Coast, but both companies are pretty small players 
there. If there is a reason this merger might get extra attention, 
it will be because Exxon and Mobil have not been terribly friendly 
toward either the Clinton administration's or the European Union's 
positions on global warming. Q. (italics)Why are you skeptical about 
the deal?(end italics) A. Well, Mobil has been trying to get bigger. 
They had talks with Amoco. They wanted to buy Conoco. But I don't 
understand where Lucio Noto, Mobil's chief executive, fits into this. 
That could be an impediment to an agreement, because in a merger I 
don't think he has a place, and he has been a very strong leader. 
Q. (italics)Mobil is the country's second-biggest oil company, behind 
Exxon. Why do they need to get bigger?(end italics) A. In the first 
decade of the next century, the really big exploration opportunities 
will be very capital intensive, and only companies with the deepest 
pockets will be able to stay in the game: Royal Dutch, Exxon and BP 
Amoco. Companies of Mobil's size are probably marginal players. Q. 
(italics)That suggests Mobil has been harder hit than Exxon by the 
downturn in prices.(end italics) A. From 1988 to 1996, Exxon's exploration 
and production expenditures rose 8 percent. Mobil's rose 14 percent. 
But Mobil's expenditures were much more sensitive to price elasticities 
of oil than Exxon's. They were pushing the envelope, and when prices 
fell they had to cut back. Exxon has tried to build a very large presence 
systematically, without paying much attention to month-to-month or 
even year-to-year fluctuations in oil prices. They are brutally efficient. 
Q. (italics)Earlier this month the Energy Department said oil prices 
would stay soft for nearly a decade. Do you agree?(end italics) A. 
You know, every time I see forecasts that go out that far I want to 
go out and buy stock in oil companies. I think we are going to see 
low oil prices for six months to a year. It is conceivable we could 
go into the next century with oil at $5 a barrel, depending on what 
happens to the world economy. During that period, we are going to 
see a substantial reduction in investment in exploration and production, 
leading to a reduction in supply coming out of non-OPEC countries. 
That will strengthen the hands of the OPEC countries. And when the 
Asian economies start growing again that will lead to a good deal 
higher oil prices, say $20 a barrel, in the next 18 months. Q. (italics)The 
number of oil companies is going to shrink in coming years, regardless, 
isn't it?(end italics) A. We are probably heading toward a world in 
which there are no more than five or six big oil companies, possibly 
eight. There is really no precedent for having as many big players 
as we have in the oil business in this modern society. Q. (italics)Do 
you think oil stocks are a good investment?(end italics) A. I think 
oil companies are still a worthwhile investment, but it is not a place 
where an investor should plan on making money over the next 9 to 12 
months. And it is an area where investors need to be careful, because 
in that period there will be a good deal of consolidation among smaller 
companies. '''

doc6='''It was new highs again for the Standard &AMP; Poor's 500-stock and 
Nasdaq composite indexes Friday as anticipation of a new wave of mergers 
and a general rush by investors to join the equity rebound pushed 
stocks up. Oil stocks led the way as investors soaked up the news 
of continuing talks between Exxon and Mobil on a merger that would 
create the world's largest oil company. Internet and computer stocks 
also rallied, helped in part by the announcement on Tuesday of America 
Online's purchase of Netscape Communications in a three-way deal involving 
Sun Microsystems. At the same time, Germany's Deutsche Bank and Bankers 
Trust are scheduled to formally announce their merger on Monday. ``There 
is no question that the merger euphoria is the headline,'' said Hugh 
Johnson, chief investment officer at the First Albany Corp. ``But 
the flow into mutual funds is also strong.'' Exxon rose 1 11/16, to 
74], while Mobil jumped 7|, to 86. Chevron, reflecting the bounce 
that other oil companies got from the merger news, climbed 5\, to 
85|. Exxon and Chevron, along with IBM, which rose 3\, to 170, were 
the main drivers of the Dow Jones industrial average. It climbed 18.80 
points, or two-tenths of a percent, to 9,333.08. It now stands just 
41 points short of the record it set Monday and up 1.9 percent for 
the week. Mobil, along with Exxon, Chevron, IBM and Microsoft, which 
rose 3 13/16, to 128 1/16, were the power behind the S &AMP; P. It 
climbed 5.46 points, or five-tenths of a percent, to 1,192.33, a new 
high, the second of the week. It jumped 2.5 percent in the last five 
trading days. Cisco Systems, up 2 15/16, to 80; MCI Worldcom, up 1 
11/16, to 62 7/16; Sun Microsystems, up 4|, to 80], and Microsoft 
pushed the Nasdaq index to its first new high since July 20. The technology-heavy 
index finished 31.23 points, or 1.57 percent, higher, at 2,016.44. 
It was up 4.6 percent for the week. Whether Friday's gains will stick 
will not be known before Monday. It was a shortened trading session, 
with the New York Stock Exchange closing at 1 p.m., and trading volume, 
at 257 million shares, made it the lightest day of the year. In the 
bond market, which also closed early because of the Thanksgiving weekend, 
the price of the 30-year Treasury bond rose 11/32, to 101 12/32. The 
bond's yield, which moves in the opposite direction from the price, 
fell to 5.16 percent from 5.18 percent on Wednesday. Long-term and 
short-term yields all slipped lower this week despite new economic 
data that indicated the economy was stronger in the third quarter 
than expected and seems to be moving along at a good pace in the current 
quarter. This small recovery in the face of stronger growth is probably 
because new inflation numbers show that prices are in check and analysts 
are still forecasting that the economy will begin to slow down next 
year. Many analysts have noted during the eight-week stock market 
rally, in which the Nasdaq composite index jumped 42 percent, that 
investors were buying again even though major financial problems around 
the world _ including a slumping Asia, a weakening Latin America and 
a troubled Russia _ have not been resolved. Johnson said he thought 
that investors, inspired by the Federal Reserve's three interest rate 
cuts in two months and by the new stimulus package in Japan, assume 
that these problems will be solved. ``Investors are looking over the 
valley and they like what they see,'' he said. But he worries that 
the financial crisis, which began in Thailand in July 1997 and was 
intensified by the effective default of Russia in August, will not 
go away quickly. ``It seems that in every financial crisis, everybody 
gets the impression that the storm has passed,'' Johnson said. ``But 
it is never that easy.``'''

doc7='''The boards of Exxon Corp. and Mobil Corp. are expected to meet Tuesday 
to consider a possible merger agreement that would form the world's 
largest oil company, a source close to the negotiations said Friday. 
The source, who spoke on condition of anonymity, said ``the prospects 
were good'' for completing an agreement. Exxon and Mobil confirmed 
Friday that they were discussing ways to combine. They cautioned, 
however, that no agreement had been reached and there was no assurance 
they would reach one. The statement sent the stock of both companies 
surging, suggesting investors believe the companies will combine. 
Shares of Exxon, the biggest U.S. oil company, rose $1.6875, or 2.3 
percent, to $74.375. Shares of Mobil, the No. 2 U.S. oil company, 
rose $7.625, or 9.7 percent, to $86. Some analysts said that if the 
two giants reached an agreement, it was likely to be in the form of 
a takeover by Exxon of Mobil. Exxon is far larger and financially 
stronger. Analysts predicted that there would be huge cuts in duplicate 
staff from both companies, which employ 122,700 people. Adam Sieminski, 
an oil analyst for BT Alex. Brown, said that the companies would probably 
make cuts to save about $3 billion to $5 billion a year. Sieminski 
and other analysts said Exxon would have to offer a premium of about 
15 to 20 percent over its price prior to Monday, when serious speculation 
of an Exxon takeover of Mobil first circulated and sent Mobil shares 
up sharply. They said the transaction would probably be an exchange 
of Mobil shares for Exxon shares. Based on Mobil's $75.25 share price 
a week ago, a takeover of the company would be worth about $70 billion. 
The merger discussions come against a backdrop of particularly severe 
pressure on Lucio Noto, the chairman, president and chief executive 
of Mobil, to find new reserves of oil and natural gas and to keep 
big projects profitable at a time of a deep decline in crude oil prices. 
``This is one of the most intelligent chief executives in the business 
and a man of considerable ability but he inherited some serious structural 
problems in his company,'' said J. Robin West, the chairman of Petroleum 
Finance Co., a consulting group to the energy industry based in Washington. 
He said that Mobil's prime assets include the Arun natural gas field 
in Indonesia, one of the largest in the world, which has contributed 
up to one-third of Mobil's profits for years but is beginning to run 
down. The field, in production since 1977, supplies liquefied natural 
gas to Japan and Korea. Although Mobil under Noto has moved quickly 
to cut costs and muscle its way into promising new areas such as Kazakhstan, 
where it is a partner in a joint venture to develop the huge Tengiz 
oil field, the payoff from such ventures is many years away. Other 
companies face similar strains. ``The challenge is to replace their 
crown jewels and grow in an increasingly competitive environment,'' 
West said. Noto has not been shy about sitting down with other companies 
such as British Petroleum and Amoco this year to see if a combination 
made sense. Although Exxon chairman Lee Raymond heads a much stronger 
and bigger company than Mobil, he has not been immune to the strains 
on the global petroleum business. Those strains intensified this year 
when Russia's economic collapse raised the risks of Exxon's extensive 
exploration venture in that country. Exxon has also been more of a 
follower than a leader in huge projects in the deep offshore fields, 
where major finds have been made near West Africa and in the Gulf 
of Mexico. '''

doc8='''Times are tough in the oil patch. Still, it boggles the mind to accept 
the notion that hardship is driving profitable Big Oil to either merge, 
as British Petroleum and Amoco have already agreed to do, or at least 
to consider the prospect, as Exxon and Mobil are doing. Oil companies 
of all stripes are getting squeezed by low petroleum prices and the 
high capital costs of exploration. Given the exotic locales of the 
most promising untapped fields, it seems unlikely that exploration 
will get cheaper. And with West Texas crude trading at around $12 
a barrel, it seems a safe bet that oil won't be selling for $100 a 
barrel by the turn of the century _ something analysts were predicting 
during the oil price run-up of the early 1980s. Philip Verleger Jr., 
publisher of Petroleum Economics Monthly and a senior adviser to the 
Brattle Group, a Cambridge, Mass., consulting firm, spent some time 
late last week talking about Mobil, Exxon and the changing dynamics 
of the oil business. Following are excerpts from the conversation: 
Q. There is a lot of focus on the antitrust aspects of an Exxon-Mobil 
deal. Do you see any problems? A. Let me say right off that I don't 
think this is a done deal. I think it is far from that. But if it 
were to happen, I don't see many problems. BP Amoco is the perfect 
end-to-end merger, one in which there is little or no overlap with 
the company you are merging with. Exxon-Mobil comes close. The first 
issue is competition in local markets. The only possible problem area 
there is on the West Coast, but both companies are pretty small players 
there. If there is a reason this merger might get extra attention, 
it will be because Exxon and Mobil have not been terribly friendly 
toward either the Clinton administration's or the European Union's 
positions on global warming. Q. Why are you skeptical about the deal? 
A. Well, Mobil has been trying to get bigger. They had talks with 
Amoco. They wanted to buy Conoco. But I don't understand where Lucio 
Noto, Mobil's chief executive, fits into this. That could be an impediment 
to an agreement, because in a merger I don't think he has a place, 
and he has been a very strong leader. Q. Mobil is the country's second-biggest 
oil company, behind Exxon. Why do they need to get bigger? A. In the 
first decade of the next century, the really big exploration opportunities 
will be very capital intensive, and only companies with the deepest 
pockets will be able to stay in the game: Royal Dutch, Exxon and BP 
Amoco. Companies of Mobil's size are probably marginal players. Q. 
That suggests Mobil has been harder hit than Exxon by the downturn 
in prices. A. From 1988 to 1996, Exxon's exploration and production 
expenditures rose 8 percent. Mobil's rose 14 percent. But Mobil's 
expenditures were much more sensitive to the price elasticities of 
oil than Exxon's. They were pushing the envelope, and when prices 
fell they had to cut back. Exxon has tried to build a very large presence 
systematically, without paying much attention to month-to-month or 
even year-to-year fluctuations in oil prices. They are brutally efficient. 
Q. Earlier this month the Energy Department said oil prices would 
stay soft for nearly a decade. Do you agree? A. You know, every time 
I see forecasts that go out that far I want to go out and buy stock 
in oil companies. I think we are going to see low oil prices for six 
months to a year. It is conceivable we could go into the next century 
with oil at $5 a barrel, depending on what happens to the world economy. 
During that period, we are going to see a substantial reduction in 
investment in exploration and production, leading to a reduction in 
supply coming out of non-OPEC countries. That will strengthen the 
hands of the OPEC countries. And when the Asian economies start growing 
again, that will lead to a good deal higher oil prices, say $20 a 
barrel, in the next 18 months. Q. The number of oil companies is going 
to shrink in coming years, regardless, isn't it? A. We are probably 
heading toward a world in which there are no more than five or six 
big oil companies, possibly eight. There is really no precedent for 
having as many big players as we have in the oil business in this 
modern society. Q. Do you think oil stocks are a good investment? 
A. I think oil companies are still a worthwhile investment, but it 
is not a place where an investor should plan on making money over 
the next 9 to 12 months. And it is an area where investors need to 
be careful, because in that period there will be a good deal of consolidation 
among smaller companies.'''

doc9='''Exxon and Mobil, the nation's two largest oil companies, confirmed 
Friday that they were discussing a possible merger, and antitrust 
lawyers, industry analysts and government officials predicted that 
any deal would require the sale of important large pieces of such 
a new corporate behemoth. Those divestitures would further reshape 
an industry already undergoing a broad transformation because of the 
low price of oil. But the mergers and other corporate combinations 
are also beginning to create a new regulatory climate among antitrust 
officials, one that may prove particularly challenging to Exxon and 
Mobil. Although the companies only confirmed that they were discussing 
the possibility of a merger, a person close to the discussions said 
the boards of both Exxon and Mobil were expected to meet Tuesday to 
consider an agreement. Shares of both surged on the New York Stock 
Exchange. Oil exploration and drilling interests would not necessarily 
present antitrust problems in an Exxon-Mobil merger because competition 
in those areas is brisk. But in retailing and marketing operations, 
an Exxon-Mobil combination would be ``like Ford merging with General 
Motors, Macy's with Gimbels,'' said Stephen Axinn, an antitrust lawyer 
in New York who represented Texaco in its acquisition of Getty more 
than a decade ago. In the United States, the deal would come under 
the purview of the Federal Trade Commission, which under the Clinton 
administration has examined large corporate mergers with a vigor not 
seen since the 1970s. The agency has blocked a number of proposed 
mergers, such as the $4 billion combination of Staples and Office 
Depot, the two largest office supply discounters, and two deals involving 
the four largest drug wholesalers. On the other hand it has approved 
other big mergers, such as Boeing's $14 billion acquisition of McDonnell 
Douglas. The agency's analysis of an Exxon-Mobil combination, a senior 
official said Friday, will turn on how it might resemble John D. Rockefeller's 
Standard Oil trust before it was dismantled by the Supreme Court in 
1911. ``The big antitrust issue is whether, by a merger or alliance, 
they will be able to get the competition of one another off their 
backs, particularly against the background of BP-Amoco and Shell-Texaco,'' 
said Eleanor M. Fox, a professor at the New York University School 
of Law and an antitrust expert. ``It may be that we are looking at 
a consolidation on the world level that looked like the consolidation 
on the national level 100 years ago.'' The Federal Trade Commission 
recently approved a significant joint venture between Shell and Texaco 
after the venture agreed to sell a refinery and divest some retailing 
operations in Hawaii and California. It is also considering the proposed 
$54 billion merger of British Petroleum and Amoco and will now have 
to re-examine that combination in light of the new talks between Exxon 
and Mobil. ``In the past, when there have been two mergers involving 
the same industry, the government has considered them together in 
deciding how to deal with them,'' said Terry Calvani, a former commissioner 
at the Federal Trade Commission whose clients now include Chevron. 
But coming in the aftermath of the Shell-Texaco and BP-Amoco deals, 
a combination of Exxon and Mobil, which have significantly overlapping 
retail and refinery businesses in the United States and Europe, poses 
questions that antitrust officials have not confronted since the 1980s. 
``It's a real test case,'' said Frederick Leuffer, a senior energy 
analyst at Bear Stearns. ``If the FTC let this one go through without 
major divesting, then everything would be fair game. Why couldn't 
GM merge with Ford?'' While the 1911 breakup of Standard Oil is viewed 
by Washington officials and industry executives as ancient history, 
Exxon and Mobil have become the dominant rivals in some retailing 
and refining markets, and in the production of lubricants and petrochemicals. 
Leuffer said he believed divestitures necessary in this case ``could 
be so large that they are deal-breakers.'' Other analysts and lawyers, 
while disagreeing that the required divestitures could kill the deal, 
said the companies would nonetheless have to shed significant operations 
and that they expected challenges to be raised by a broad spectrum 
of constituents, including competitors, customers and state officials. 
``These are not absolute obstacles,'' said John Hervey, an analyst 
at Donaldson Lufkin &AMP; Jenrette. In the past, when two big oil 
companies have merged, aggressive attorneys general from the states 
have almost always become involved in raising questions because of 
the high visibility of the local gas station. ``When a state attorney 
general drives by and sees four stations on a corner and two of them 
are Mobil and Exxon, they are certain to raise questions,'' Axinn 
said. Exxon and Mobil have significant concentrations of retailing 
and marketing concerns in the Northeast, the Southwest and the West 
Coast, where they also have big refining operations. Because the two 
companies are involved in everything from exploration and shipping 
to refining and retailing, the meaning of the deal for consumers will 
take months for the regulators to sort out. The regulators examining 
such a transaction would dissect each business, determine whether 
its market is global, national or more local, and determine whether 
the combined entity has too high a concentration of the business in 
those areas. Experts agreed that the regulators are most likely to 
permit Exxon and Mobil to keep their exploratory and oil production 
businesses because those areas are already highly competitive and 
a merger would not result in higher prices. Hervey said those markets 
are already so fragmented that the combined market share of major 
American and European oil companies is only 17 percent. '''

doc10='''They have been downsized, cut back and re-engineered. So when the 
900 or so remaining blue-collar workers here at Mobil's largest domestic 
refinery, out of about 1,500 a decade ago, heard last week that their 
company was discussing a possible merger with Exxon, it was like a 
siren warning them that an already suspect valve might be about to 
blow. ``I think it's a terrible thing,'' said Dick Mabry, a refinery 
operator, as he emerged in the plant's artificial twilight from the 
main gate after his 12-hour shift ended at 4:30 on Sunday morning. 
He stopped to rub eyes rimmed with red, but on this topic his bedtime 
could be delayed. ``It's a revival of the Standard Oil Company. It's 
going to put 20 or 30 thousand people out of work. I think the Justice 
Department should step in and stop it.'' Ernest Lewis, whose overalls 
bore a ``Big E'' patch appropriate to his scale, added his uh-huh's. 
The latest evidence of where things were heading hulked right nearby, 
he said, glancing over at a new power plant likely to be operated 
by an outside company without the unions that now man the refinery's 
generators. But if the Mobil Corp. has to be sold, Lewis said, noting 
the gains in his company stock holdings, a buyer as solid and large 
as the Exxon Corp. might be the least of all evils. ``If we merged 
with Chevron, we'd be Moron,'' he added. A growing American economy 
that can make a billionaire out of someone with an unproven idea for 
Internet marketing is still sloughing off workers in older industries, 
in petroleum as much as any. Those here point to the tote board by 
the Beaumont plant's brick headquarters, that they say shows they 
have already handled 171 million barrels out of 130 million planned 
for the full year. But the numbers that matter even more are the ones 
like 89.9, 88.9, even 81.9, on nearby gas stations _ the lowest prices, 
after inflation, since the Depression. Which is why Mobil and Exxon 
are considering combining into the world's largest oil company. Some 
people close to the talks cautioned that no deal would be considered 
by their boards until at least Tuesday, maybe Wednesday. And that 
is why, beginning last Wednesday evening, the phone at the home of 
Jimmy Herrington, the president of the Oil, Chemical and Atomic Workers 
Local 4-243 rang without stop. No, he didn't know about that merger 
talk on television, said Herrington, who also works full-time producing 
lubricants. He had asked some Mobil managers, in a meeting earlier 
this month, about all the rumors, but they said they had heard nothing. 
Oil industry analysts say that the first targets of a combined company's 
efforts to cut billions of dollars in annual costs would be the office 
staff and the professionals, like geologists and engineers, in the 
field. One company's accountants could almost do the work for two. 
But the crews here fully expect that an Exxon or any other buyer would 
ask yet again whether the refinery could turn more crude oil into 
gasoline, motor oil and other products with even fewer people. Union 
leaders raise the prospect that Exxon would have to sell the refinery. 
Antitrust regulators, they say, are bound to notice that Exxon has 
refineries an hour's drive in one direction and three hours in the 
other, along a Gulf of Mexico crescent that forms the petrochemical 
industry's home. The Beaumont plant, a steaming, humming chemistry 
set lining the Neches River off the Gulf, has become the prime provider 
of livelihoods here since it was built almost under the spray of the 
nearby Spindletop gusher. With mounting overtime that can stretch 
a shift to 16 hours or more, workers regularly make $55,000 or $65,000 
a year. ``People go there to retire there,'' Herrington said, as he 
drove around the plant's fenced periphery. Lewis, in his 17th year, 
is a third-generation employee. But his nephew laboring here too is 
the exception. The workers streaming to and from the plant before 
dawn are mostly balding or going gray, a sign that for a full generation 
the refinery has been more concerned about how to get rid of workers 
than how to attract them. The cutbacks have, so far, come through 
attrition, with retirements often encouraged by incentives. But the 
plywood sheets covering the windows of most of the fast food places 
and gas stations around Herrington's union hall advertise that the 
best times are long gone. His members chafe at the experts who come 
in from Wall Street to question the justification for every person's 
job. In tiring and dangerous tasks, they question the elimination 
of most relief laborers in favor of covering vacations and sickness 
with overtime (although some like the extra pay, and they say the 
plant has become safer over the years). They complain about the growing 
numbers of outside contractors taking over formerly unionized tasks. 
But with many workers choosing to invest at least some retirement 
savings in Mobil stock, a 1990s ethos is gaining. Some share credit 
with the plant's management for the efficiency measures they agree 
are necessary for true job security. Some take the attitude that every 
company is always for sale. ``They will not be too concerned about 
what we feel about it,'' said Sam Salim, one of the electrical plant 
workers whose future is uncertain. ``But if they fork out $60 billion? 
I'd look it over.'' With most Mobil executives saying as little as 
possible for now, calls on Sunday to the local plant manager and a 
company spokesman did not elicit a peep. Union leaders, however, are 
already squawking. ''I don't believe creating new monopolies is the 
way to prop up the industry,'' said Robert Wages, a former refinery 
operator himself and now the union's president, by telephone Saturday. 
Nevertheless, with admirable foresight the union negotiated a clause 
in last November's three-year contract extension guaranteeing that 
any company buyer would have to keep to its terms. Many members, who 
typically came to work after high school, are already molding the 
oil companies' latest exploits into case studies fit for rapacious 
MBA's. ``They're the biggest,'' said Bobby Whisneant, an assistant 
operator in the gasoline and lubrication oil units, referring to Exxon. 
He was coming, early on Sunday morning, through a plant gate whose 
white canopy seems borrowed from a self-service station. ``So they 
go buy the second biggest. That's one way to get rid of the competition. 
I just hope it's not something like the 80s _ buying companies and 
scrapping them.'' Or something like the Robber Baron era a century 
before, said Mabry, another operator. ``Didn't the teachers teach 
us all through school that the Standard Oil Company would never come 
back? Remember that?'' he said, looking to his friend, Lewis, the 
Big E, for agreement. ``But I better shut up. I still work for Mobil.'' 
``Used to,'' Lewis said. '''

body=doc1+doc2+doc3+doc4+doc5+doc6+doc7+doc8+doc9+doc10



#summarize(doc2)
model = Summarizer()
###embedding+kmeans result
print('\n\n1st stage\n\n')
result = model(body, min_length=60)
full = ''.join(result)

f = open('projects/test-summarization2/system/task1_englishSyssum1.txt','w')
print (full,file=f)
f.close()





#perform centroid word embedding
print('\n\n2nd stage\n\n')
result2=summarize(full)
#print('\n\n2nd stage\n\n')
#print(result2)
f = open('projects/test-summarization2/system/task1_englishSyssum2.txt','w')
print (result2,file=f)
f.close()





print('\n\n3rd stage\n\n')
result3=MMRsummarize(result2)
#print(result3)
f = open('projects/test-summarization2/system/task1_englishSyssum3.txt','w')
print (result3,file=f)
f.close()




