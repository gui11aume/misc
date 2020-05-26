BASEURL= ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/008/SRR10564808

all: SRR10564808_1.fastq.gz SRR10564808_2.fastq.gz

SRR10564808_1.fastq.gz:
	wget $(BASEURL)/SRR10564808_1.fastq.gz
SRR10564808_2.fastq.gz:
	wget $(BASEURL)/SRR10564808_2.fastq.gz


#GSM4197251 	E14 cells, AC_6xPCR_24h_rep1
#GSM4197252 	E14 cells, AC_6xPCR_24h_rep2
#GSM4197253 	E14 cells, AC_6xPCR_24h_rep3
#GSM4197254 	E14 cells, AC_6xPCR_24h_rep4        SAMN13430868 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/026/SRR10564726/SRR10564726_1.fastq.gz
#GSM4197255 	E14 cells, AC_6xPCR_48h_rep1        SAMN13430867 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/027/SRR10564727/SRR10564727_1.fastq.gz
#GSM4197256 	E14 cells, AC_6xPCR_48h_rep2        SAMN13430866 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/028/SRR10564728/SRR10564728_1.fastq.gz
#GSM4197257 	E14 cells, AC_6xPCR_48h_rep3        SAMN13430865 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/029/SRR10564729/SRR10564729_1.fastq.gz
#GSM4197258 	E14 cells, AC_6xPCR_48h_rep4        SAMN13430864 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/008/SRR10564808/SRR10564808_1.fastq.gz
#GSM4197259 	E14 cells, AC_6xPCR_no_ISceI_rep1   SAMN13430863
#GSM4197260 	E14 cells, AC_6xPCR_no_ISceI_rep2   SAMN13430862 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/010/SRR10564810/SRR10564810_1.fastq.gz
#GSM4197261 	E14 cells, AC_6xPCR_no_ISceI_rep3   SAMN13430861 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/011/SRR10564811/SRR10564811_1.fastq.gz
#GSM4197262 	E14 cells, AC_6xPCR_no_ISceI_rep4   SAMN13430860 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/012/SRR10564812/SRR10564812_1.fastq.gz
#GSM4197263 	E14 cells, AC_6xPCR_no_ISceI_rep5   SAMN13430847 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/013/SRR10564813/SRR10564813_1.fastq.gz
#GSM4197264 	E14 cells, AC_6xPCR_no_ISceI_rep6   SAMN13430846 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/014/SRR10564814/SRR10564814_1.fastq.gz
#GSM4197265 	E14 cells, AC_6xPCR_no_ISceI_rep7   SAMN13430845
#GSM4197266 	E14 cells, AC_6xPCR_no_ISceI_rep8   SAMN13430844
#GSM4197267 	E14 cells, AC_6xPCR_no_ISceI_rep9   SAMN13430843
#GSM4197268 	E14 cells, AC_LA_24h_rep1           SAMN13430842 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/018/SRR10564818/SRR10564818_1.fastq.gz
#GSM4197269 	E14 cells, AC_LA_24h_rep2           SAMN13430888 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/019/SRR10564819/SRR10564819_1.fastq.gz
#GSM4197270 	E14 cells, AC_LA_24h_rep3           SAMN13430887 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/020/SRR10564820/SRR10564820_1.fastq.gz
#GSM4197271 	E14 cells, AC_LA_24h_rep4           SAMN13430886 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/021/SRR10564821/SRR10564821_1.fastq.gz
#GSM4197272 	E14 cells, AC_LA_48h_rep1           SAMN13430885 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/022/SRR10564822/SRR10564822_1.fastq.gz
#GSM4197273 	E14 cells, AC_LA_48h_rep2           SAMN13430884 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/023/SRR10564823/SRR10564823_1.fastq.gz
#GSM4197274 	E14 cells, AC_LA_48h_rep3           SAMN13430883 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/024/SRR10564824/SRR10564824_1.fastq.gz
#GSM4197275 	E14 cells, AC_LA_48h_rep4           SAMN13430882 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/025/SRR10564825/SRR10564825_1.fastq.gz
#GSM4197276 	E14 cells, AC_LA_no_ISceI_rep1      SAMN13430841 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/026/SRR10564826/SRR10564826_1.fastq.gz
#GSM4197277 	E14 cells, AC_LA_no_ISceI_rep2      SAMN13430840 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/027/SRR10564827/SRR10564827_1.fastq.gz
#GSM4197278 	E14 cells, AC_LA_no_ISceI_rep3      SAMN13430839 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/028/SRR10564828/SRR10564828_1.fastq.gz
#GSM4197279 	E14 cells, AC_LA_no_ISceI_rep4      SAMN13430838 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/029/SRR10564829/SRR10564829_1.fastq.gz
#GSM4197280 	E14 cells, AG_6xPCR_24h_rep1        SAMN13430837 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/030/SRR10564830/SRR10564830_1.fastq.gz
#GSM4197281 	E14 cells, AG_6xPCR_24h_rep2        SAMN13430836 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/031/SRR10564831/SRR10564831_1.fastq.gz
#GSM4197282 	E14 cells, AG_6xPCR_24h_rep3        SAMN13430835 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/032/SRR10564832/SRR10564832_1.fastq.gz
#GSM4197283 	E14 cells, AG_6xPCR_24h_rep4        SAMN13430834 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/033/SRR10564833/SRR10564833_1.fastq.gz
#GSM4197284 	E14 cells, AG_6xPCR_48h_rep1        SAMN13430833 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/034/SRR10564834/SRR10564834_1.fastq.gz
#GSM4197285 	E14 cells, AG_6xPCR_48h_rep2        SAMN13430832 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/035/SRR10564835/SRR10564835_1.fastq.gz
#GSM4197286 	E14 cells, AG_6xPCR_no_ISceI_rep1   SAMN13430831 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/036/SRR10564836/SRR10564836_1.fastq.gz
#GSM4197287 	E14 cells, AG_6xPCR_no_ISceI_rep2   SAMN13430830 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/037/SRR10564837/SRR10564837_1.fastq.gz
#GSM4197288 	E14 cells, AG_6xPCR_no_ISceI_rep3   SAMN13430829 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/038/SRR10564838/SRR10564838_1.fastq.gz
#GSM4197289 	E14 cells, AG_6xPCR_no_ISceI_rep4   SAMN13430828 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/039/SRR10564839/SRR10564839_1.fastq.gz
#GSM4197290 	E14 cells, AG_6xPCR_no_ISceI_rep5   SAMN13430827 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/040/SRR10564840/SRR10564840_1.fastq.gz
#GSM4197291 	E14 cells, AG_6xPCR_no_ISceI_rep6   SAMN13430826 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/041/SRR10564841/SRR10564841_1.fastq.gz
#GSM4197292 	E14 cells, AG_6xPCR_no_ISceI_rep7   SAMN13430825 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/042/SRR10564842/SRR10564842_1.fastq.gz
#GSM4197293 	E14 cells, AG_LA_24h_rep1           SAMN13430824 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/043/SRR10564843/SRR10564843_1.fastq.gz
#GSM4197294 	E14 cells, AG_LA_24h_rep2           SAMN13430823 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/044/SRR10564844/SRR10564844_1.fastq.gz
#GSM4197295 	E14 cells, AG_LA_24h_rep3           SAMN13430822 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/045/SRR10564845/SRR10564845_1.fastq.gz
#GSM4197296 	E14 cells, AG_LA_24h_rep4           SAMN13430821 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/046/SRR10564846/SRR10564846_1.fastq.gz
#GSM4197297 	E14 cells, AG_LA_48h_rep1           SAMN13430820
#GSM4197298 	E14 cells, AG_LA_48h_rep2           SAMN13430872 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/048/SRR10564848/SRR10564848_1.fastq.gz
#GSM4197299 	E14 cells, AG_LA_48h_rep3           SAMN13430881 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/049/SRR10564849/SRR10564849_1.fastq.gz
#GSM4197300 	E14 cells, AG_LA_48h_rep4           SAMN13430880
#GSM4197301 	E14 cells, AG_LA_no_ISceI_rep1      SAMN13430879
#GSM4197302 	E14 cells, AG_LA_no_ISceI_rep2      SAMN13430878 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/052/SRR10564852/SRR10564852_1.fastq.gz
#GSM4197303 	E14 cells, AG_LA_no_ISceI_rep3      SAMN13430877
#GSM4197304 	E14 cells, AG_LA_no_ISceI_rep4      SAMN13430876 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/054/SRR10564854/SRR10564854_1.fastq.gz
#GSM4197305 	E14 cells, AG_LA_no_ISceI_rep5      SAMN13430875 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/055/SRR10564855/SRR10564855_1.fastq.gz
#GSM4197306 	E14 cells, AG_LA_no_ISceI_rep6      SAMN13430874 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/056/SRR10564856/SRR10564856_1.fastq.gz
#GSM4197307 	E14 cells, AG_LA_no_ISceI_rep7      SAMN13430904 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/057/SRR10564857/SRR10564857_1.fastq.gz
#GSM4197308 	E14 cells, AG_LA_no_ISceI_rep8      SAMN13430873 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/058/SRR10564858/SRR10564858_1.fastq.gz
#GSM4197309 	E14 cells, CA_6xPCR_24h_rep1        SAMN13430908
#GSM4197310 	E14 cells, CA_6xPCR_24h_rep2        SAMN13430907 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/060/SRR10564860/SRR10564860_1.fastq.gz
#GSM4197311 	E14 cells, CA_6xPCR_24h_rep3        SAMN13430906 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/061/SRR10564861/SRR10564861_1.fastq.gz
#GSM4197312 	E14 cells, CA_6xPCR_24h_rep4        SAMN13430905 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/062/SRR10564862/SRR10564862_1.fastq.gz
#GSM4197313 	E14 cells, CA_6xPCR_48h_rep1        SAMN13430903 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/063/SRR10564863/SRR10564863_1.fastq.gz
#GSM4197314 	E14 cells, CA_6xPCR_48h_rep2        SAMN13430902
#GSM4197315 	E14 cells, CA_6xPCR_48h_rep3        SAMN13430901
#GSM4197316 	E14 cells, CA_6xPCR_48h_rep4        SAMN13430900
#GSM4197317 	E14 cells, CA_6xPCR_no_ISceI_rep1   SAMN13430899
#GSM4197318 	E14 cells, CA_6xPCR_no_ISceI_rep2   SAMN13430898 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/056/SRR10564756/SRR10564756_1.fastq.gz
#GSM4197319 	E14 cells, CA_6xPCR_no_ISceI_rep3   SAMN13430897
#GSM4197320 	E14 cells, CA_6xPCR_no_ISceI_rep4   SAMN13430896
#GSM4197321 	E14 cells, CA_LA_24h_rep1           SAMN13430895
#GSM4197322 	E14 cells, CA_LA_24h_rep2           SAMN13430894
#GSM4197323 	E14 cells, CA_LA_24h_rep3           SAMN13430893 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/061/SRR10564761/SRR10564761_1.fastq.gz
#GSM4197324 	E14 cells, CA_LA_24h_rep4           SAMN13430892 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/062/SRR10564762/SRR10564762_1.fastq.gz
#GSM4197325 	E14 cells, CA_LA_48h_rep1           SAMN13430891 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/063/SRR10564763/SRR10564763_1.fastq.gz
#GSM4197326 	E14 cells, CA_LA_48h_rep2           SAMN13430890 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/064/SRR10564764/SRR10564764_1.fastq.gz
#GSM4197327 	E14 cells, CA_LA_48h_rep3           SAMN13430889 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/065/SRR10564765/SRR10564765_1.fastq.gz
#GSM4197328 	E14 cells, CA_LA_48h_rep4           SAMN13430930 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/066/SRR10564766/SRR10564766_1.fastq.gz
#GSM4197329 	E14 cells, CA_LA_no_ISceI_rep1      SAMN13430929 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/067/SRR10564767/SRR10564767_1.fastq.gz
#GSM4197330 	E14 cells, CA_LA_no_ISceI_rep2      SAMN13430928 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/068/SRR10564768/SRR10564768_1.fastq.gz
#GSM4197331 	E14 cells, CA_LA_no_ISceI_rep3      SAMN13430927 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/069/SRR10564769/SRR10564769_1.fastq.gz
#GSM4197332 	E14 cells, CA_LA_no_ISceI_rep4      SAMN13430926 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/070/SRR10564770/SRR10564770_1.fastq.gz
#GSM4197333 	E14 cells, TC_6xPCR_24h_rep1        SAMN13430925
#GSM4197334 	E14 cells, TC_6xPCR_24h_rep2        SAMN13430924 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/072/SRR10564772/SRR10564772_1.fastq.gz
#GSM4197335 	E14 cells, TC_6xPCR_24h_rep3        SAMN13430923 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/073/SRR10564773/SRR10564773_1.fastq.gz
#GSM4197336 	E14 cells, TC_6xPCR_24h_rep4        SAMN13430918
#GSM4197337 	E14 cells, TC_6xPCR_48h_rep1        SAMN13430917 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/075/SRR10564775/SRR10564775_1.fastq.gz
#GSM4197338 	E14 cells, TC_6xPCR_48h_rep2        SAMN13430922 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/076/SRR10564776/SRR10564776_1.fastq.gz
#GSM4197339 	E14 cells, TC_6xPCR_48h_rep3        SAMN13430921 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/077/SRR10564777/SRR10564777_1.fastq.gz
#GSM4197340 	E14 cells, TC_6xPCR_48h_rep4        SAMN13430920 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/078/SRR10564778/SRR10564778_1.fastq.gz
#GSM4197341 	E14 cells, TC_6xPCR_no_ISceI_rep1   SAMN13430919
#GSM4197342 	E14 cells, TC_6xPCR_no_ISceI_rep2   SAMN13430916
#GSM4197343 	E14 cells, TC_6xPCR_no_ISceI_rep3   SAMN13430915
#GSM4197344 	E14 cells, TC_6xPCR_no_ISceI_rep4   SAMN13430914 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/082/SRR10564782/SRR10564782_1.fastq.gz
#GSM4197345 	E14 cells, TC_6xPCR_no_ISceI_rep5   SAMN13430913 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/083/SRR10564783/SRR10564783_1.fastq.gz
#GSM4197346 	E14 cells, TC_6xPCR_no_ISceI_rep6   SAMN13430912 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/084/SRR10564784/SRR10564784_1.fastq.gz
#GSM4197347 	E14 cells, TC_6xPCR_no_ISceI_rep7   SAMN13430911 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/085/SRR10564785/SRR10564785_1.fastq.gz
#GSM4197348 	E14 cells, TC_6xPCR_no_ISceI_rep8   SAMN13430910 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/086/SRR10564786/SRR10564786_1.fastq.gz
#GSM4197349 	E14 cells, TC_LA_24h_rep1           SAMN13430909 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/087/SRR10564787/SRR10564787_1.fastq.gz
#GSM4197350 	E14 cells, TC_LA_24h_rep2           SAMN13430946 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/088/SRR10564788/SRR10564788_1.fastq.gz
#GSM4197351 	E14 cells, TC_LA_24h_rep3           SAMN13430945 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/089/SRR10564789/SRR10564789_1.fastq.gz
#GSM4197352 	E14 cells, TC_LA_24h_rep4           SAMN13430944 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/090/SRR10564790/SRR10564790_1.fastq.gz
#GSM4197353 	E14 cells, TC_LA_48h_rep1           SAMN13430943
#GSM4197354 	E14 cells, TC_LA_48h_rep2           SAMN13430942
#GSM4197355 	E14 cells, TC_LA_48h_rep3           SAMN13430941
#GSM4197356 	E14 cells, TC_LA_48h_rep4           SAMN13430940
#GSM4197357 	E14 cells, TC_LA_no_ISceI_rep1      SAMN13430939
#GSM4197358 	E14 cells, TC_LA_no_ISceI_rep2      SAMN13430938
#GSM4197359 	E14 cells, TC_LA_no_ISceI_rep3      SAMN13430937
#GSM4197360 	E14 cells, TC_LA_no_ISceI_rep4      SAMN13430936
#GSM4197361 	E14 cells, TC_LA_no_ISceI_rep5      SAMN13430935
#GSM4197362 	E14 cells, TC_LA_no_ISceI_rep6      SAMN13430934 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/000/SRR10564800/SRR10564800_1.fastq.gz
#GSM4197363 	E14 cells, TC_LA_no_ISceI_rep7      SAMN13430933
#GSM4197364 	E14 cells, TC_LA_no_ISceI_rep8      SAMN13430932 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/002/SRR10564802/SRR10564802_1.fastq.gz
#GSM4197365 	E14 cells, TG_6xPCR_24h_rep1        SAMN13430931 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/003/SRR10564803/SRR10564803_1.fastq.gz
#GSM4197366 	E14 cells, TG_6xPCR_24h_rep2        SAMN13430964 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/004/SRR10564804/SRR10564804_1.fastq.gz
#GSM4197367 	E14 cells, TG_6xPCR_24h_rep3        SAMN13430963 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/005/SRR10564805/SRR10564805_1.fastq.gz
#GSM4197368 	E14 cells, TG_6xPCR_24h_rep4        SAMN13430968 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/006/SRR10564806/SRR10564806_1.fastq.gz
#GSM4197369 	E14 cells, TG_6xPCR_48h_rep1        SAMN13430967 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/007/SRR10564807/SRR10564807_1.fastq.gz
#GSM4197370 	E14 cells, TG_6xPCR_48h_rep2        SAMN13430966
#GSM4197371 	E14 cells, TG_6xPCR_48h_rep3        SAMN13430965 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/031/SRR10564731/SRR10564731_1.fastq.gz
#GSM4197372 	E14 cells, TG_6xPCR_48h_rep4        SAMN13430962
#GSM4197373 	E14 cells, TG_6xPCR_no_ISceI_rep1   SAMN13430961
#GSM4197374 	E14 cells, TG_6xPCR_no_ISceI_rep2   SAMN13430960
#GSM4197375 	E14 cells, TG_6xPCR_no_ISceI_rep3   SAMN13430959
#GSM4197376 	E14 cells, TG_6xPCR_no_ISceI_rep4   SAMN13430958
#GSM4197377 	E14 cells, TG_LA_24h_rep1           SAMN13430957
#GSM4197378 	E14 cells, TG_LA_24h_rep2           SAMN13430956
#GSM4197379 	E14 cells, TG_LA_24h_rep3           SAMN13430955
#GSM4197380 	E14 cells, TG_LA_24h_rep4           SAMN13430954 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/040/SRR10564740/SRR10564740_1.fastq.gz
#GSM4197381 	E14 cells, TG_LA_48h_rep1           SAMN13430953
#GSM4197382 	E14 cells, TG_LA_48h_rep2           SAMN13430952 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/042/SRR10564742/SRR10564742_1.fastq.gz
#GSM4197383 	E14 cells, TG_LA_48h_rep3           SAMN13430951
#GSM4197384 	E14 cells, TG_LA_48h_rep4           SAMN13430950
#GSM4197385 	E14 cells, TG_LA_no_ISceI_rep1      SAMN13430949 ftp://ftp.sra.ebi.ac.uk/vol1/fastq/SRR105/045/SRR10564745/SRR10564745_1.fastq.gz
#GSM4197386 	E14 cells, TG_LA_no_ISceI_rep2      SAMN13430948
#GSM4197387 	E14 cells, TG_LA_no_ISceI_rep3      SAMN13430979
#GSM4197388 	E14 cells, TG_LA_no_ISceI_rep4      SAMN13430969
#GSM4197389 	E14 cells, iPCR_1                   SAMN13430978
#GSM4197390 	E14 cells, iPCR_2                   SAMN13430977
#GSM4197391 	E14 cells, iPCR_3                   SAMN13430976
