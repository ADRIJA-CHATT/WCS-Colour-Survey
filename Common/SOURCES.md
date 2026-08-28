# Data and methodological sources

## Primary WCS sources

1. World Color Survey. **WCS Data Archives.** University of California, Berkeley.  
   https://linguistics.berkeley.edu/wcs/data.html
   
   Documents the WCS design, 110 languages, the 330 standardized Munsell chips, and the public data files.

2. World Color Survey. **`term.txt` README.**  
   https://linguistics.berkeley.edu/wcs/data/readme/term-readme.html
   
   Documents the four naming-response fields: language number, speaker number, chip number, and colour-term abbreviation.

3. World Color Survey. **`spkr-lsas.txt` README.**  
   https://linguistics.berkeley.edu/wcs/data/20100912/spkr-lsas.html
   
   Documents speaker metadata and the fact that speaker numbers are unique within language.

4. J. Vosten, maintained WCS data mirror.  
   https://github.com/jvosten/wcs/tree/master/data-raw
   
   The setup script uses the public raw mirror for reproducible acquisition of `term.txt`, `spkr-lsas.txt`, and the CIELAB mapping, with historical Berkeley URLs as fallbacks.

## Colour-language and psycholinguistic background

5. Berlin, B. & Kay, P. (1969). *Basic Color Terms: Their Universality and Evolution*. University of California Press.

6. Kay, P. & Regier, T. (2003). Resolving the question of color naming universals. *Proceedings of the National Academy of Sciences*, 100(15), 9085--9089. DOI: 10.1073/pnas.1532837100.

7. Regier, T., Kay, P., & Khetarpal, N. (2007). Color naming reflects optimal partitions of color space. *Proceedings of the National Academy of Sciences*, 104(4), 1436--1441. DOI: 10.1073/pnas.0610341104.

8. Winawer, J., Witthoft, N., Frank, M. C., Wu, L., Wade, A. R., & Boroditsky, L. (2007). Russian blues reveal effects of language on color discrimination. *Proceedings of the National Academy of Sciences*, 104(19), 7780--7785. DOI: 10.1073/pnas.0701644104.

9. Roberson, D., Davidoff, J., Davies, I. R. L., & Shapiro, L. R. (2005). Color categories: Evidence for the cultural relativity hypothesis. *Cognitive Psychology*, 50(4), 378--411. DOI: 10.1016/j.cogpsych.2004.10.001.

## Statistical methodology

10. Liang, K.-Y. & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika*, 73(1), 13--22. DOI: 10.1093/biomet/73.1.13.

11. Papke, L. E. & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. *Journal of Applied Econometrics*, 11(6), 619--632. DOI: 10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1.

12. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics*, 7(1), 1--26. DOI: 10.1214/aos/1176344552.

13. Miller, G. A. (1955). Note on the bias of information estimates. In H. Quastler (Ed.), *Information Theory in Psychology II-B*, 95--100. Free Press.

14. Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory* (2nd ed.). Wiley.
