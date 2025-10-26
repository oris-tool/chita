# CHITA Library

This repository provides CHITA, a Java library implementing a quantitative approach to predict the spread of infectious diseases within a cluster. 

CHITA is presented in a paper titled "An observation-based quantitative approach to predict the spread of infectious diseases within a cluster", authored by Laura Carnevali, Silvia Dani, Niccolò Niccoli, Benedetta Picano, and Enrico Vicario, currently submitted for a journal publication. 

The most distinctive features of CHITA are: 
- definition of a custom-made extensible metamodel of an infection chain of a disease; 
- automated translation of a metamodel instance into a Stochastic Time Petri Net (STPN) characterizing the disease evolution from contact to infectiousness;
- implementation of an efficient quantitative approach to predict the spread of infectious diseases within a cluster, exploiting not only the STPN model of disease evolution in an individual but also observations of contacts, symptoms, and results of diagnostic tests;
- randomly generated data sets of observations of contacts, symptoms, and results of diagnostic tests.

## Experimental reproducibility

## Installation

This repository provides a ready-to-use Maven project that you can easily import into an Eclipse workspace to start working with the [CHITA library](https://github.com/oris-tool/chita/) (the version `2.0.0-SNAPSHOT` of the [Sirio library](https://github.com/oris-tool/sirio) is included as a Maven dependency). Just follow these steps:

1. **Install Java >= 11.** For Windows, you can download a [package from Oracle](https://www.oracle.com/java/technologies/downloads/#java11); for Linux, you can run `apt-get install openjdk-11-jdk`; for macOS, you can run `brew install --cask java`. 

2. **Download Eclipse.** The [Eclipse IDE for Java Developers](http://www.eclipse.org/downloads/eclipse-packages/) package is sufficient.

3. **Clone this project.** Inside Eclipse:
   - Select `File > Import > Maven > Check out Maven Projects from SCM` and click `Next`.
   - If the `SCM URL` dropbox is grayed out, click on `m2e Marketplace` and install `m2e-egit`. You will have to restart Eclipse.
   - As `SCM URL`, type: `git@github.com:oris-tool/chita.git` and click `Next` and then `Finish`.
