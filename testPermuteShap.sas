
* program to test permutation on y procedure for multiple regression;
libname lib '.';

proc printto new print='junk.out';
proc printto new log='junk.log';

data allstats;  * to store final results;
data allp;      * to store final results;
%macro sim(nreps);
%do rep = 1 %to %sysevalf(&nreps);

* simulate observed dataset;
data a;
  b1 = 0.4;
  b2 = 0.2;
  b3 = 0;
  b4 = 0;
  sigma = 1;
  junkid=1;
  * retain seed 123479;
  retain seed 0;
  do id = 1 to 200;
     if ranuni(seed) < 0.5 then x1=1; else x1=0;
     x2 = rannor(seed);
     if ranuni(seed) < 0.5 then x3=1; else x3=0;
     x4 = rannor(seed);
     y = b1*x1 + b2*x2 + b3*x3 + b4*x4 + sigma*rannor(seed);
     output;
  end;
  drop seed;

* compute means and deviations for use in shap calcs.  Same values can be used in permuted data;
proc means data=a; by junkid;
  var x1-x4 y;
  output out=j mean=m1 m2 m3 m4 my stddev=s1 s2 s3 s4 sy;
data j; set j;
  drop _type_;
data a;  merge a j; by junkid;
  dx1 = abs(x1 - m1);
  dx2 = abs(x2 - m2);
  dx3 = abs(x3 - m3);
  dx4 = abs(x4 - m4);

proc means data=a; by junkid;
  var dx1-dx4;
  output out=j2 mean=md1 md2 md3 md4;

data j; merge j j2;

proc reg data=a outest=k;
  model y = x1 x2 x3 x4 / pcorr2;
  ods output parameterestimates = p;
run;

data o; set p;
  retain t1 t2 t3 t4;
  source = 'obs';
  if variable='x1' then t1=tvalue*tvalue;
  else if variable='x2' then t2=tvalue*tvalue;
  else if variable='x3' then t3=tvalue*tvalue;
  else if variable='x4' then t4=tvalue*tvalue;
  if variable='x4' then output;
  keep source t1-t4;
data p2;
  retain rsq1 rsq2 rsq3 rsq4;
  keep rsq1-rsq4;
  do j = 1 to 5;
    set p;
    if variable='x1' then rsq1=SqPartCorrTypeII;
    if variable='x2' then rsq2=SqPartCorrTypeII;
    if variable='x3' then rsq3=SqPartCorrTypeII;
    if variable='x4' then rsq4=SqPartCorrTypeII;
  end;

data k; set k;
  rename x1=b1 x2=b2 x3=b3 x4=b4;
  drop _type_;
data kp; merge k p2;

* compute shap;
data kj; merge kp j;
  * on the squared scale;
  shap1= (b1*s1/sy)*(b1*s1/sy);
  shap2= (b2*s2/sy)*(b2*s2/sy);
  shap3= (b3*s3/sy)*(b3*s3/sy);
  shap4= (b4*s4/sy)*(b4*s4/sy);
  source = 'obs';

data obs; merge o kj;
run;

data b; set a;
  yp = y;
  pid = _N_;
proc sort; by pid;

data sall;

%macro proces;
%do i = 1 %to 1000;
    data c; set b;
      retain seed 0;
      u = ranuni(seed);
      keep u yp;
    proc sort; by u;
    data c; set c;
      pid = _N_;
    data d; merge b c; by pid;
      y = yp;

    proc reg data=d outest=k;
      model y = x1 x2 x3 x4 / pcorr2;
      ods output parameterestimates = p;
    * store tvalues from regression for perm test of standard regression t stats;
    data s; set p;
      retain t1 t2 t3 t4;
      if variable='x1' then t1=tvalue*tvalue;
      else if variable='x2' then t2=tvalue*tvalue;
      else if variable='x3' then t3=tvalue*tvalue;
      else if variable='x4' then t4=tvalue*tvalue;
      if variable='x4' then output;
      keep t1-t4;
    * compute and store shap values based on regression for perm test of shap values;

    * compute shap;
    data p2;
      retain rsq1 rsq2 rsq3 rsq4;
      keep rsq1-rsq4;
      do j = 1 to 5;
        set p;
        if variable='x1' then rsq1=SqPartCorrTypeII;
        if variable='x2' then rsq2=SqPartCorrTypeII;
        if variable='x3' then rsq3=SqPartCorrTypeII;
        if variable='x4' then rsq4=SqPartCorrTypeII;
      end;
    proc print;

    data k; set k;
      rename x1=b1 x2=b2 x3=b3 x4=b4;
      drop _type_;
    data kp; merge k p2;
    proc print;

    data kj; merge kp j;
      * rep = %sysevalf(&rep);
      /*
      * on the sqrt scale;
      shap1= abs(b1*s1/sy);
      shap2= abs(b2*s2/sy);
      shap3= abs(b3*s3/sy);
      shap4= abs(b4*s4/sy);
      */
      * on the squared scale;
      shap1= (b1*s1/sy)*(b1*s1/sy);
      shap2= (b2*s2/sy)*(b2*s2/sy);
      shap3= (b3*s3/sy)*(b3*s3/sy);
      shap4= (b4*s4/sy)*(b4*s4/sy);
      * source = 'sim';

    data sim; merge s kj;
    * proc print;
    run;
    * proc print;
    data sall; set sall sim;

%end;
%mend;
%proces;

data sall; set sall;
  if t1 ne .;
  source='sim';

data all; set obs sall;
  drop  _MODEL_  _DEPVAR_   _RMSE_  Intercept _Type_ _Freq_ y;
proc sort; by descending t1;

%macro getp(stat);
%do i = 1 %to 4;
  proc sort data=all; by descending &stat.&i;
  data &stat.p&i; set all end=eof;
    retain junkid obs&stat.&i &stat.rank&i num;
    if _N_=1 then num=0;
    junkid=1;
    if source='obs' then do;
       obs&stat.&i=&stat.&i;
       &stat.rank&i=_N_;
    end;
    num=num+1;
    if eof then do;
       &stat.p&i = &stat.rank&i/num;
       output;
    end;
%end;
%mend;
%getp(t)
%getp(shap);

data stats; merge tp1 tp2 tp3 tp4 shapp1 shapp2 shapp3 shapp4;
  drop junkid;
  rep = %sysevalf(&rep);
data ponly; merge tp1 tp2 tp3 tp4 shapp1 shapp2 shapp3 shapp4;
  keep rep num tp1-tp4 shapp1-shapp4 obst1-obst4 obsshap1-obsshap4;
  rep = %sysevalf(&rep);

data allp; set allp ponly;
data allstats; set allstats stats;
%end;
%mend;
%sim(5);

proc printto;

proc print data=allp;
run;

data lib.allp; set allp;
data lib.allstats; set allstats;

data allp; set allp;
  if tp1 ne .;
data allstats; set allstats;
  if tp1 ne .;
/*
proc print data=allp;
run;
proc print data=allstats;
run;
  */
data allp; set allp;
array obst obst1-obst4;
array obstp obsp1-obsp4;
array t tp1-tp4;
array shap shapp1-shapp4;
array logt logtp1-logtp4;
array logshap logshapp1-logshapp4;
array logobst logobsp1-logobsp4;
array rejobst rejobst1-rejobst4;
array rejt rejt1-rejt4;
array rejshap rejshap1-rejshap4;
do i = 1 to 4;
   obstp[i] = 1-probchi(obst[i],1);
   if obsp < 0.05 then rejobst[i] = 1; else rejobst[i]=0;
   if t[i] < 0.05 then rejt[i] = 1; else rejt[i]=0;
   if shap[i] < 0.05 then rejshap[i] = 1; else rejshap[i]=0;
   logt[i] = log10(t[i]);
   logshap[i] = log10(shap[i]);
   logobst[i] = log10(obstp[i]);

end;
proc print data=allp;
title 'allp after pval computations';
run;

proc means data=allp;
  var rejt1-rejt4 rejshap1-rejshap4 rejobst1-rejobst4 ;
run;
proc print data=allp;
  var obst1-obst4;
run;

proc gplot data=allp;
  symbol1 v=star i=rl;
  plot
   logshapp1*logtp1=1
   logshapp2*logtp2=1
   logshapp3*logtp3=1
   logshapp4*logtp4=1
   logobsp1*logtp1=1
   logobsp2*logtp2=1
   logobsp3*logtp3=1
   logobsp4*logtp4=1
  ;
run;
