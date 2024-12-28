* -- allkids13.sas -- do a run w/ no traffic on 4 yrs from 11 to 15 to
     get variance of 4 yr growth over this age span on log scale for
     use in power calculations;

 /*
options nocenter ls=70;
libname lib '.';

*libname lib '.';

data a;
 *set lib.alltd_3knot;
 *set lib.knots3;
  set lib.allkids_center10;
  if townname='Lompoc' or townname='Lake Gregory' then dist_1=10;
  d1 = dist_1;
  d1q1 = 0; d1q2=0; d1q3=0; d1q4=0;
  if d1 ne . then do;
     if d1 < 0.5 then do;
       d1q1 = 1;
       d1cat=1;
     end;
     else if d1 < 1.0 then do;
       d1q2=1;
       d1cat=2;
     end;
     else if d1 < 1.5 then do;
       d1q3=1;
       d1cat=3;
     end;
     else do;
       d1q4=1;
       d1cat=4;
     end;
  end;
  if d1 ne . then do;
     if d1 < 0.25 then do;
       d2cat=1;
     end;
     else if d1 < 0.5 then do;
       d2cat=2;
     end;
     else if d1 < 1.0 then do;
       d2cat=3;
     end;
     else if d1 < 1.5 then do;
       d2cat=4;
     end;
     else do;
       d2cat=5;
     end;
  end;
  if townname='Lompoc' or townname='Lake Gregory' then do;
     d1cat=4;
     d2cat=5;
     d1q4=1;
  end;
  d1catlin = d1cat;
  d2catlin = d2cat;
run;

proc sort data=a; by id;
data sub; set a; by id;
  if first.id;
proc sort; by townname;
proc means data=sub; by townname;
  var d1catlin d2catlin d1q1 d1q2 d1q3 d1q4;
  output out=junk mean=meand1catlin meand2catlin meand1q1 meand1q2
            meand1q3 meand1q4;
proc sort data=a; by townname;
data a; merge junk a; by townname;
dd1catlin = d1catlin - meand1catlin;
dd2catlin = d2catlin - meand2catlin;
dd1q1 = d1q1 - meand1q1;
dd1q2 = d1q2 - meand1q2;
dd1q3 = d1q3 - meand1q3;
dd1q4 = d1q4 - meand1q4;
run;



run;
   */
/*
data b; set a; if year >=1994 and year <= 1998;
 lfe = log(fev);
 tstar = (agepft-11)/3;
run;
proc print data=b (obs=100);
  var id year agepft t tstar;
run;
proc freq;
  tables cohort*year;
run;
proc means;
  class year male;
  var agepft fev;
run;
  */
run;
data allest;
%macro mixrun(x);
%macro rundat(dat);
* -- 3 knot model;
proc mixed data=&dat; *where pft_counts>7;
  class id townname race hisp cohort ftc d1cat;
  model lfe = tstar
             townname townname*tstar
             rht rht*rht male*rht male*rht*rht
             rbmi rbmi*rbmi male*rbmi male*rbmi*rbmi
             ri ttasthma male*ttasthma smokyear
             race hisp yasthma cohort male
             yasthma*tstar cohort*tstar male*tstar
             male*race male*hisp male*yasthma
             &x
             /solution; * outp=pred residual;
*
  rht rht*rht male*rht male*rht*rht
             rbmi rbmi*rbmi male*rbmi male*rbmi*rbmi
             ri male*ri ttasthma male*ttasthma ftc exer smokyear
             race hisp yasthma cohort male
             race*t hisp*t yasthma*t cohort*t male*t
             race*tau1 hisp*tau1 yasthma*tau1 cohort*tau1 male*tau1
             race*tau2 hisp*tau2 yasthma*tau2 cohort*tau2 male*tau2
             race*tau3 hisp*tau3 yasthma*tau3 cohort*tau3 male*tau3
             male*race male*hisp male*yasthma;
  random intercept t / type=un subject=id*townname;
*  random intercept t / type=un subject=townname;
/*
  random tau1;
  random tau2;
  random tau3;
 */
 ods output solutionf=j;

 data k; set j;
   length datset $3.;
   if effect="t*&x" or effect="&x" or effect='mno2' or effect='t*mno2';
   model='re: x t*x   ';
   datset = "dat= &dat";
/*proc mixed data=a;
  class id townname race hisp cohort ftc;
  model mef = t tau1 tau2 tau3
             rht rht*rht male*rht male*rht*rht
             rbmi rbmi*rbmi male*rbmi male*rbmi*rbmi
             ri male*ri ttasthma male*ttasthma ftc exer smokyear
             race hisp yasthma cohort male
             race*t hisp*t yasthma*t cohort*t male*t
             race*tau1 hisp*tau1 yasthma*tau1 cohort*tau1 male*tau1
             race*tau2 hisp*tau2 yasthma*tau2 cohort*tau2 male*tau2
             race*tau3 hisp*tau3 yasthma*tau3 cohort*tau3 male*tau3
             male*race male*hisp male*yasthma
             t* &x /solution; * outp=pred residual;
 random intercept t tau1 tau2 tau3 / type=simple subject=id*townname;
 random intercept t tau1 tau2 tau3 / type=simple subject=townname;
 ods output solutionf=jj;

 data kk; set jj;
   if effect="t* &x";
   model='re: only t*x';*/

 data allest; set allest k; *kk;

%mend rundat;
%rundat(b);
*%rundat(b);
*%rundat(c);
*%rundat(d);
*%rundat(f);
*%rundat(m);
%mend;
%mixrun();
run;
/*
proc means data=sub n p25 median p75 qrange;
  where townname ne 'Lake Gregory' and townname ne 'Lompoc';
  var dlogdist1;
run;

/*

%mixrun(d1cat t*d1cat);
%mixrun(d1q1 d1q2 d1q3 t*d1q1 t*d1q2 t*d1q3);
%mixrun(dd1q1 dd1q2 dd1q3 t*dd1q1 t*dd1q2 t*dd1q3);
%mixrun(dd1catlin t*dd1catlin);
%mixrun(dd2catlin t*dd2catlin);
run;
/*
%mixrun(d2cat);
%mixrun(d1catlin);
%mixrun(d2catlin);
run;
/*
%mixrun(ddist1);
%mixrun(ddist4);
%mixrun(dno2fwy);
%mixrun(dno2nfwy);
%mixrun(dno2tot);
%mixrun(dnoxtot);
%mixrun(no2fwy);
%mixrun(pm25ec_fwy);
%mixrun(pm25oc_fwy);
%mixrun(pm25ec_nfwy);
%mixrun(pm25oc_nfwy);
%mixrun(dlogdist1);
%mixrun(dlogdist4);
%mixrun(dlogno2fwy);
%mixrun(dlogno2nfwy);
%mixrun(d1catlin);  * town centered version of d1cat where d1cat has values 1, 2, 3, 4;
%mixrun(d1cat);
%mixrun(d1cat_v1);  * town centered version of d1cat where d1cat has values 1, 2, 3, 4;
%mixrun(d1cat_v2);  * town centered version of d1cat where d1cat has values 1, 2, 3, 4;
run;


data allest; set allest; if estimate ne .;
 drop townname cohort hisp race ftc df;
proc sort; by effect datset;
proc print data=allest;
run;
*/
/*
data a; set a;
  junkid=1;
proc sort data=a; by id;
data sub; set a; by id;
  if first.id;
  if townname='Lompoc' or townname='Lake Gregory' then delete;
  no2_tot = no2_fwy + no2_nfwy;
  pm25ec_tot = pm25ec_fwy + pm25ec_nfwy;
proc means p25 p50 p75 n;
  var no2_fwy no2_nfwy no2_tot pm25ec_fwy pm25ec_nfwy pm25ec_tot;
run;
%macro quart(poll);
%macro which(typ);
proc means data=sub; by junkid;
  var &poll._&typ;
  output out=j p25=p25 p50=p50 p75=p75;
data a; merge j a; by junkid;
  if &poll._&typ ne . then do;
     if &poll._&typ < p25      then &poll.&typ.cat=1;
     else if &poll._&typ < p50 then &poll.&typ.cat=2;
     else if &poll._&typ < p75 then &poll.&typ.cat=3;
     else                           &poll.&typ.cat=4;
  end;
  if townname='Lompoc' or townname='Lake Gregory' then &poll.&typ.cat=1;
  &poll.&typ.catlin = &poll.&typ.cat;
%mend which;
%which(fwy);
%which(nfwy);
%which(tot);
%mend quart;
%quart(no2);
run;
proc print data=a;
  var townname id no2_fwy no2fwycat no2fwycatlin;
run;
proc freq data=a;
  tables townname*no2fwycat;
run;
  */
















/*
proc means data=a;
  where year=1997;
  class male;
  var fev;
run;
  */
run;
