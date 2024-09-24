time = [0.0,0.08080808080808081,0.16161616161616163,0.24242424242424243,0.32323232323232326,0.4040404040404041,0.48484848484848486,0.5656565656565657,0.6464646464646465,0.7272727272727273,0.8080808080808082,0.888888888888889,0.9696969696969697,1.0505050505050506,1.1313131313131315,1.2121212121212122,1.292929292929293,1.373737373737374,1.4545454545454546,1.5353535353535355,1.6161616161616164,1.696969696969697,1.777777777777778,1.8585858585858588,1.9393939393939394,2.0202020202020203,2.101010101010101,2.181818181818182,2.262626262626263,2.3434343434343434,2.4242424242424243,2.505050505050505,2.585858585858586,2.666666666666667,2.747474747474748,2.8282828282828287,2.909090909090909,2.98989898989899,3.070707070707071,3.151515151515152,3.2323232323232327,3.3131313131313136,3.393939393939394,3.474747474747475,3.555555555555556,3.6363636363636367,3.7171717171717176,3.7979797979797985,3.878787878787879,3.95959595959596,4.040404040404041,4.121212121212122,4.202020202020202,4.282828282828283,4.363636363636364,4.444444444444445,4.525252525252526,4.606060606060606,4.686868686868687,4.767676767676768,4.848484848484849,4.92929292929293,5.01010101010101,5.090909090909092,5.171717171717172,5.252525252525253,5.333333333333334,5.414141414141414,5.494949494949496,5.575757575757576,5.6565656565656575,5.737373737373738,5.818181818181818,5.8989898989899,5.97979797979798,6.060606060606061,6.141414141414142,6.222222222222222,6.303030303030304,6.383838383838384,6.464646464646465,6.545454545454546,6.626262626262627,6.707070707070708,6.787878787878788,6.868686868686869,6.94949494949495,7.030303030303031,7.111111111111112,7.191919191919193,7.272727272727273,7.353535353535354,7.434343434343435,7.515151515151516,7.595959595959597,7.676767676767677,7.757575757575758,7.838383838383839,7.91919191919192,8.0];
signal1D = [0.0,0.08072016411790384,0.16091351669367898,0.24005668436454086,0.3176331476874822,0.39313661214832984,0.4660743124388942,0.5359702284371984,0.6023681919020362,0.6648348636063459,0.7229625614794606,0.7763719213006606,0.8247143725793011,0.8676744134629433,0.9049716698265449,0.9363627251042848,0.9616427089218196,0.9806466341609311,0.9932504737303577,0.9993719700153771,0.9989711717233568,0.9920506946216033,0.978655704465837,0.9588736222307036,0.9328335535661072,0.9007054462029555,0.8626989808074191,0.8190622025224275,0.7700799021274968,0.7160717573820826,0.6573902466827755,0.5944183486506397,0.5275670426620876,0.4572726266358116,0.38399386958094966,0.30820901749007645,0.2304126721177387,0.15111256301485315,0.07082623388594546,-0.009922335104641916,-0.09060614703340818,-0.17069862760691001,-0.24967706179272697,-0.3270260052633736,-0.4022406483887269,-0.47483011082223986,-0.5443206451791693,-0.6102587288983565,-0.6722140241088524,-0.729782186184138,-0.7825875026542022,-0.830285345252909,-0.8725644190976193,-0.9091487943220443,-0.9397997069030881,-0.9643171169287782,-0.9825410141374196,-0.9943524622074992,-0.9996743749829051,-0.9984720195675029,-0.9907532430056771,-0.9765684210694334,-4.78005064743148,-4.6460626987684694,-4.48175271741497,-4.288193054940258,-4.066646957837897,-3.818560323082243,-3.545552261636025,-3.2494045314953564,-2.932049909236172,-2.59555957595407,-2.242129599921763,-1.8740666041837208,-1.493772712626891,-1.1037298727753124,-0.7064836576243404,-0.3046266522299522,0.09921846652110244,0.502416046532275,0.9023346617995591,1.2963642861326226,1.6819333272124086,2.056525409814106,2.4176957986617067,2.7630873537320335,3.090445913877697,3.3976350083694395,3.6826498003444614,3.943630171160412,4.178872860261297,4.386842581325865,4.566182042150027,4.715720802869666,4.834482914711594,4.921693289419119,4.976782757782894,4.999391784263061,4.989372813459598,4.946791233116909];

% time = linspace(0, 99, 100);
% signal1D = rand([100, 1]);

signal1D_approx = ifft(fft(signal1D));
% scatter(time, signal1D)

scatter(time, signal1D_approx - signal1D, 'r*')