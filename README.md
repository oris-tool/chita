# CHITA Library

This repository provides CHITA, a Java library implementing a quantitative approach to predict the spread of infectious diseases within a cluster. CHITA is presented in a paper titled "An observation-based quantitative approach to predict the spread of infectious diseases within a cluster", authored by Laura Carnevali, Silvia Dani, Niccolò Niccoli, Benedetta Picano, and Enrico Vicario, currently submitted for a journal publication. 

The most distinctive features of CHITA are: 
- definition of a custom-made extensible metamodel of an infection chain of a disease; 
- automated translation of a metamodel instance into a Stochastic Time Petri Net (STPN) characterizing the disease evolution from contact to infectiousness;
- implementation of an efficient quantitative approach to predict the spread of infectious diseases within a cluster, exploiting not only the STPN model of disease evolution in an individual but also observations of contacts, symptoms, and results of diagnostic tests;
- randomly generated data sets of observations of contacts, symptoms, and results of diagnostic tests.

## Experimental reproducibility

To support reproducibility of the experimental results reported in the paper, perform the steps reported below to repeat the experiments.

1. Run the simulation
   ```powershell
   python run_n_simulations.py
   ```

   This script generates datasets and intermediate output used by the analysis.

2. Run the Java STPN analysis 
   
   Import the project into Eclipse or IntelliJ and run the `main` method in `com.chita.analysis.STPNAnalysis`.
  
   Note: The analysis writes `stpn_solution.csv` if it does not exist.

3. Run the simulation for 10 steps

   ```powershell
   python run_n_simultations.py --rep 10
   ```

4. Plot results

   ```powershell
   python plot_results.py
   python plot_results_csv.py
   ```


## Installation

This repository provides a ready-to-use Maven project that you can easily import into an Eclipse workspace to start working with the [CHITA library](https://github.com/oris-tool/chita/) (the version `2.0.0-SNAPSHOT` of the [Sirio library](https://github.com/oris-tool/sirio) is included as a Maven dependency). Just follow these steps:

1. **Install Java >= 11.** For Windows, you can download a [package from Oracle](https://www.oracle.com/java/technologies/downloads/#java11); for Linux, you can run `apt-get install openjdk-11-jdk`; for macOS, you can run `brew install --cask java`. 

2. **Download Eclipse or IntelliJ.** Regarding Eclipse, the [Eclipse IDE for Java Developers](http://www.eclipse.org/downloads/eclipse-packages/) package is sufficient.

3. **Clone this project.** Inside Eclipse:
   - Select `File > Import > Maven > Check out Maven Projects from SCM` and click `Next`.
   - If the `SCM URL` dropbox is grayed out, click on `m2e Marketplace` and install `m2e-egit`. You will have to restart Eclipse.
   - As `SCM URL`, type: `git@github.com:oris-tool/chita.git` and click `Next` and then `Finish`.

4. **Install Python >= 3.8.** We provide a minimal quick-start for the Python tools. First choose one for your platform:
   - Windows:

   ```powershell
   python -m venv .venv
   . .\.venv\Scripts\Activate.ps1 
   pip install -r requirements.txt
   ```

   - macOS / Linux:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

   - If a shell blocks script execution on Windows, run:

   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

   - Python dependencies are listed in `requirements.txt`.

## Licence

CHITA is released under the [GNU Affero General Public License v3.0](https://choosealicense.com/licenses/agpl-3.0).
