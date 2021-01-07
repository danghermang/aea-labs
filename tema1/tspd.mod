 
 
 using CP;
 
 
 execute{
 }
 
 int n = ...;
 range i = 0..n;
 range k = 0..n+1;
 dvar interval visit[i];
 dvar interval tVisit[k]; // vizitat de camion
 dvar interval dVisit[i]; // vizitat de drone
 range j = 0..n;
 dvar interval tdVisit[i][j]; // drona care pleaca de la nodul j al camionului catre nodul i al dronei
 dvar interval dtVisit[i][j]; // drona care pleaca de la nodul i al dronei catre nodul j al camionului
 dvar interval dVisit_before[i]; // timpul dus drona
 dvar interval dVisit_after[i]; // timp intors drona
 int dist[i][j] = ...;
 tuple triplet {int id1; int id2; int value;};
 {triplet} w = {<i,j,dist[i][j]> | i in 0..n, j in 0..n};
 dvar sequence tVisitSeq in tVisit;
 dvar interval aux[0..1]; 
 
 minimize
 	endOf(tVisit[n]);
 subject to {
 	forall (i in 0..n) presenceOf(visit[i]);
 	presenceOf(tVisit[0]);
 	presenceOf(tVisit[n+1]);
 	first(tVisitSeq, tVisit[0]);
 	last(tVisitSeq, tVisit[n]);
 	noOverlap(tVisitSeq, w);
 	noOverlap(dVisit);
 	forall(i in 0..n) {
 		alternative(visit[i], append(tVisit[i],dVisit[i]));
 		alternative(dVisit_before[i], all (j in 0..n) tdVisit[i][j]);
 		alternative(dVisit_after[i], all (j in 0..n) dtVisit[i][j]);
 		span(dVisit[i], append(dVisit_before[i], dVisit_after[i]));
 		endAtStart(dVisit_before[i], dVisit_after[i]);
 		presenceOf(dVisit[i]) => presenceOf(dVisit_before[i]) && presenceOf(dVisit_after[i]);
 	}
 	forall(i, j in 0..n) {
 		presenceOf(tdVisit[i][j]) => presenceOf(tVisit[j]);	
 		presenceOf(dtVisit[i][j]) => presenceOf(tVisit[j]);
 		startBeforeStart(tVisit[j], tdVisit[i][j]);
 		startBeforeEnd(tdVisit[i][j], tVisit[j]);
 		endBeforeEnd(dtVisit[i][j], tVisit[j]);
 	}
 }