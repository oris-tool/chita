
/* This program is part of the ORIS Tool.
  * Copyright (C) 2011-2025 The ORIS Authors.
  *
  * This program is free software: you can redistribute it and/or modify
  * it under the terms of the GNU Affero General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * This program is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU Affero General Public License for more details.
  *
  * You should have received a copy of the GNU Affero General Public License
  * along with this program.  If not, see <https://www.gnu.org/licenses/>.
  */

package com.chita.analysis;

import com.google.gson.JsonObject;
import org.oristool.models.pn.Priority;
import org.oristool.models.stpn.MarkingExpr;
import org.oristool.models.stpn.TransientSolution;
import org.oristool.models.stpn.trans.TreeTransient;
import org.oristool.models.stpn.trees.StochasticTransitionFeature;
import org.oristool.petrinet.Marking;
import org.oristool.petrinet.PetriNet;
import org.oristool.petrinet.Place;
import org.oristool.petrinet.Transition;

import java.io.*;
import java.math.BigDecimal;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;


public class STPNAnalysis {
    // 1. Build model
    // 2. Set Marking
    // 3. Run analysis
    // 4. Get Solution

    public static <R, S> TransientSolution<R, S> buildModel(int samples, float step){
        PetriNet net = new PetriNet();
        Marking marking = new Marking();

        Place Asymptomatic = net.addPlace("Asymptomatic");
        Place DevelopingSymptoms = net.addPlace("DevelopingSymptoms");
        Place EffectiveContact = net.addPlace("EffectiveContact");
        Place Healed = net.addPlace("Healed");
        Place Infected = net.addPlace("Infected");
        Place Infectious = net.addPlace("Infectious");
        Place Isolated = net.addPlace("Isolated");
        Place Symptomatic = net.addPlace("Symptomatic");
        Place Symptomatology = net.addPlace("Symptomatology");
        Place p0 = net.addPlace("p0");
        Place p1 = net.addPlace("p1");
        Place p2 = net.addPlace("p2");
        Place p3 = net.addPlace("p3");
        Place p4 = net.addPlace("p4");
        Place p5 = net.addPlace("p5");
        Place p6 = net.addPlace("p6");
        Place p7 = net.addPlace("p7");
        Transition effectiveContact = net.addTransition("effectiveContact");
        Transition noSymptoms = net.addTransition("noSymptoms");
        Transition symptoms = net.addTransition("symptoms");
        Transition t0 = net.addTransition("t0");
        Transition t1 = net.addTransition("t1");
        Transition t10 = net.addTransition("t10");
        Transition t11 = net.addTransition("t11");
        Transition t12 = net.addTransition("t12");
        Transition t13 = net.addTransition("t13");
        Transition t14 = net.addTransition("t14");
        Transition t15 = net.addTransition("t15");
        Transition t2 = net.addTransition("t2");
        Transition t3 = net.addTransition("t3");
        Transition t5 = net.addTransition("t5");
        Transition t6 = net.addTransition("t6");
        Transition t7 = net.addTransition("t7");
        Transition t8 = net.addTransition("t8");
        Transition t9 = net.addTransition("t9");

        //Generating Connectors
        net.addPostcondition(symptoms, DevelopingSymptoms);
        net.addPrecondition(p7, t14);
        net.addPrecondition(p3, t11);
        net.addPrecondition(DevelopingSymptoms, t0);
        net.addPrecondition(Symptomatology, symptoms);
        net.addPrecondition(p5, t9);
        net.addPostcondition(t7, p3);
        net.addPrecondition(Symptomatology, noSymptoms);
        net.addPrecondition(Infectious, t5);
        net.addPostcondition(t1, p1);
        net.addPostcondition(t3, Symptomatic);
        net.addPostcondition(t6, Healed);
        net.addPostcondition(t10, p7);
        net.addPostcondition(t2, Symptomatic);
        net.addPostcondition(effectiveContact, Symptomatology);
        net.addPrecondition(Asymptomatic, t5);
        net.addPostcondition(t13, Infectious);
        net.addPostcondition(t12, p5);
        net.addPostcondition(t8, p4);
        net.addPrecondition(p2, t6);
        net.addPostcondition(t15, Isolated);
        net.addPostcondition(t5, p2);
        net.addPostcondition(t11, p5);
        net.addPrecondition(Symptomatic, t15);
        net.addPrecondition(Infected, t8);
        net.addPrecondition(Infectious, t15);
        net.addPrecondition(EffectiveContact, effectiveContact);
        net.addPostcondition(effectiveContact, Infected);
        net.addPrecondition(p1, t3);
        net.addPrecondition(Infected, t7);
        net.addPrecondition(DevelopingSymptoms, t1);
        net.addPostcondition(t9, p6);
        net.addPrecondition(p0, t2);
        net.addPostcondition(t14, Infectious);
        net.addPostcondition(t0, p0);
        net.addPrecondition(p4, t12);
        net.addPrecondition(p5, t10);
        net.addPrecondition(p6, t13);
        net.addPostcondition(noSymptoms, Asymptomatic);

        //Generating Properties
        marking.setTokens(Asymptomatic, 0);
        marking.setTokens(DevelopingSymptoms, 0);
        marking.setTokens(EffectiveContact, 1);
        marking.setTokens(Healed, 0);
        marking.setTokens(Infected, 0);
        marking.setTokens(Infectious, 0);
        marking.setTokens(Isolated, 0);
        marking.setTokens(Symptomatic, 0);
        marking.setTokens(Symptomatology, 0);
        marking.setTokens(p0, 0);
        marking.setTokens(p1, 0);
        marking.setTokens(p2, 0);
        marking.setTokens(p3, 0);
        marking.setTokens(p4, 0);
        marking.setTokens(p5, 0);
        marking.setTokens(p6, 0);
        marking.setTokens(p7, 0);
        effectiveContact.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("1", net)));
        effectiveContact.addFeature(new Priority(0));
        noSymptoms.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.35", net)));
        noSymptoms.addFeature(new Priority(0));
        symptoms.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.65", net)));
        symptoms.addFeature(new Priority(0));
        t0.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.81", net)));
        t0.addFeature(new Priority(0));
        t1.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.19", net)));
        t1.addFeature(new Priority(0));
        t10.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.11", net)));
        t10.addFeature(new Priority(0));
        t11.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("0.6958 / 24", net)));
        t12.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("0.1626 / 24", net)));
        t13.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("1.357/24", net)));
        t14.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("0.170/24", net)));
        t15.addFeature(StochasticTransitionFeature.newUniformInstance(new BigDecimal("0"), new BigDecimal("24")));
        t2.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("0.6958 / 24", net)));
        t3.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("0.1626 / 24", net)));
        t5.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("1/(10.68 * 24)", net)));
        t6.addFeature(StochasticTransitionFeature.newExponentialInstance(new BigDecimal("1"), MarkingExpr.from("1/(1.27 * 24)", net)));
        t7.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.81", net)));
        t7.addFeature(new Priority(0));
        t8.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.19", net)));
        t8.addFeature(new Priority(0));
        t9.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("0.89", net)));
        t9.addFeature(new Priority(0));

        // Run analysis
        TreeTransient analysis = TreeTransient.builder()
                .greedyPolicy(new BigDecimal(samples), BigDecimal.ZERO)
                .timeStep(new BigDecimal(step))
                .build();

        TransientSolution<Marking, Marking> result = analysis.compute(net, marking);

        var rewardRates = TransientSolution.rewardRates("Infectious");
        var rewardedSolution = TransientSolution.computeRewards(false, result, rewardRates);

        return (TransientSolution<R, S>) rewardedSolution;




    }


    public static void fillWithGranularity(HashMap<Double, Double> map) {
        HashMap<Double, Double> filledMap = new HashMap<>();
        Double[] keys = map.keySet().toArray(new Double[0]);
        Arrays.sort(keys);

        for (int i = 0; i < keys.length - 1; i++) {
            double start = keys[i];
            double end = keys[i + 1];
            double startValue = map.get(start);
            double endValue = map.get(end);
            double step = (endValue - startValue) / (end - start);


            for (double j = start; j <= end; j++) {
                double key = Math.round(j * 10.0) / 10.0; // Round to one decimal place
                filledMap.put(key, startValue + (j - start) * step);
            }
        }

        // Add the last key-value pair
        filledMap.put(keys[keys.length - 1], map.get(keys[keys.length - 1]));

        // Replace the original map with the filled map
        map.clear();
        map.putAll(filledMap);
    }

    public static void main(String[] args) throws Exception {
        float time_step = 0.1f;


        List<String> jsonFiles = new ArrayList<>();


        Path startPath = Path.of(".");
        // Recursively search this folder and subfolders for files ending with "simulated.json"
        try (java.util.stream.Stream<Path> walk = Files.walk(startPath)) {
            walk.filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith("simulated.json"))
                .forEach(p -> {
                    String fp = p.toString();
                    if (!jsonFiles.contains(fp)) jsonFiles.add(fp);
                });
        } catch (IOException e) {
            e.printStackTrace();
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(startPath)) {
            for (Path entry : stream) {
                if (Files.isRegularFile(entry) && entry.toString().endsWith("simulated.json")) {
                    jsonFiles.add(entry.toString());
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        jsonFiles.forEach(System.out::println);
        HashMap<Double, Double> phi = new HashMap<>(); // curve of relevance of symptoms
        // the key is in DAYS * 24 (hours)
        phi.put(0.0, 0.0);
        phi.put(2.5 * (int)(Math.round(24.0 / time_step)), 0.65 * 0.75);
        phi.put(4.0 * (int)(Math.round(24.0 / time_step)), 0.65 * 0.95);
        phi.put(10.0 * (int)(Math.round(24.0 / time_step)), 0.65 * 0.75);
        phi.put(15.0 * (int)(Math.round(24.0 / time_step)), 0.65 * 0.5);
        phi.put(20.0 * (int)(Math.round(24.0 / time_step)), 0.65 * 0.25);
        phi.put(25.0 * (int)(Math.round(24.0 / time_step)), 0.65 * 0.125);
        phi.put(30.0 * (int)(Math.round(24.0 / time_step)), 0.65 * 0.0625);
        phi.put(35.0 * (int)(Math.round(24.0 / time_step)), 0.0);
        // fill
        fillWithGranularity(phi);

        HashMap<Double, Double> theta = new HashMap<>(); // curve of relevance of tests
        // the key is in DAYS * 24 (hours)
        theta.put(0.0, 0.0);
        theta.put(2.5 * (int)(Math.round(24.0 / time_step)), 0.75);
        theta.put(4.0 * (int)(Math.round(24.0 / time_step)), 0.95);
        theta.put(14.0 * (int)(Math.round(24.0 / time_step)), 0.75);
        theta.put(19.0 * (int)(Math.round(24.0 / time_step)), 0.5);
        theta.put(24.0 * (int)(Math.round(24.0 / time_step)), 0.25);
        theta.put(29.0 * (int)(Math.round(24.0 / time_step)), 0.125);
        theta.put(34.0 * (int)(Math.round(24.0 / time_step)), 0.0625);
        theta.put(35.0 * (int)(Math.round(24.0 / time_step)), 0.0);
        // fill
        fillWithGranularity(theta);
        ArrayList<double[]> priorsCombinations = new ArrayList<>();
        priorsCombinations.add(new double[]{0.5, 0.5, 0.5});

        int time_limit = 84;

        HashMap<Integer, Double> stpnSolutionMap = new HashMap<>();
        File stpnCsvFile = new File("stpn_solution.csv");

// Check if the solution file already exists.
        if (!stpnCsvFile.exists()) {
            // --- 1. File NOT found: Generate, Save, and Populate Map ---
            System.out.println("stpn_solution.csv not found. Generating new solution...");

            // Generate the solution by running the STPN analysis.
            TransientSolution<Integer, Double> solution = buildModel(time_limit * 24, time_step);

            try (FileWriter writer = new FileWriter(stpnCsvFile)) {
                writer.append("Time,State,Value\n"); // CSV Header

                for (int i = 0; i < solution.getSolution().length; i++) {
                    // Write the primary state (j=0) to the file and map.
                    double value = solution.getSolution()[i][0][0];
                    writer.append(i + ",0," + value + "\n");
                    stpnSolutionMap.put(i, value);
                }
                writer.flush();
                System.out.println("STPN solution saved to " + stpnCsvFile.getName());
            } catch (IOException e) {
                System.err.println("Error writing STPN solution to CSV file.");
                e.printStackTrace();
            }
        } else {
            // --- 2. File FOUND: Load from CSV into the Map ---
            System.out.println("Loading existing solution from stpn_solution.csv...");
            try (BufferedReader reader = new BufferedReader(new FileReader(stpnCsvFile))) {
                String line;
                reader.readLine(); // Skip header row

                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(",");
                    if (parts.length == 3) {
                        int time = Integer.parseInt(parts[0]);
                        int state = Integer.parseInt(parts[1]);
                        double value = Double.parseDouble(parts[2]);

                        // Only load data for the primary state (state=0) into our map.
                        if (state == 0) {
                            stpnSolutionMap.put(time, value);
                        }
                    }
                }
                System.out.println("Successfully loaded " + stpnSolutionMap.size() + " time points.");
            } catch (IOException | NumberFormatException e) {
                System.err.println("Error reading STPN solution from CSV file. Consider deleting it to regenerate.");
                e.printStackTrace();
            }
        }

        List<Long> timesList = new ArrayList<>();
        for (int repetition = 0; repetition < 1; repetition++) {
            for (int documentId = 0; documentId < jsonFiles.size(); documentId++) {
                int n_iterations = 3;
                double[] priorsValues = {0.5, 0.5, 0.5};
                String filePath = jsonFiles.get(documentId);
                String documentName = filePath.substring(filePath.lastIndexOf("\\") + 1, filePath.lastIndexOf("."));
                JsonObject jsonObject = JsonFileReader.readJsonFromFile(filePath);
                int n_subjects = 0;

                Queue<Event> events = new LinkedList<>();
                Random random = new Random();
                if (jsonObject != null) {
                    n_subjects = jsonObject.get("n_subjects").getAsInt();
                    time_limit = jsonObject.get("time_limit").getAsInt();
                    for (int i = 0; i < jsonObject.getAsJsonArray("events").size(); i++) {
                        JsonObject event = jsonObject.getAsJsonArray("events").get(i).getAsJsonObject();
                        String[] involved_subjects = new String[event.get("involved_subjects").getAsJsonArray().size()];
                        for (int j = 0; j < event.get("involved_subjects").getAsJsonArray().size(); j++) {
                            involved_subjects[j] = event.get("involved_subjects").getAsJsonArray().get(j).getAsString();
                        }
                        Double riskFactor = event.has("risk_factor") && !event.get("risk_factor").isJsonNull() ? event.get("risk_factor").getAsDouble() : null;
                        Boolean result = event.has("result") && !event.get("result").isJsonNull() ? event.get("result").getAsBoolean() : null;
                        events.add(new Event(event.get("type").getAsString(),
                                involved_subjects,
                                event.get("time").getAsInt(),
                                riskFactor,
                                result));
                    }
                } else {
                    throw new Exception("Failed to read JSON from file.");
                }
                int time_horizon = (int) (Math.round(time_limit * 24.0 / time_step));

                // Create the names based on n_subjects
                String[] names = new String[n_subjects];
                for (int i = 0; i < n_subjects; i++) {
                    names[i] = String.valueOf(i + 1);
                }


                if (n_iterations > n_subjects || n_iterations <= 0) throw new AssertionError();
                HashMap<Integer, Tracks> tracks_record = new HashMap<>();

                // Here we iterate over the symptoms and tests to gather that information
                ArrayList<Event> symptomsAndTests = new ArrayList<>();
                for (Event event : events) {
                    if (((event.type.equals("Symptoms") && event.result.equals(true)) || event.type.equals("Test")) && event.time < time_limit) { // we assume that the symptoms are only for the subject involved in the external contact
                        symptomsAndTests.add(event);
                    }
                }

                for (int current_iteration = 0; current_iteration < n_iterations; current_iteration++) {
                    HashMap<String, double[]> probabilityOfNotBeingInfectedDueToPreviousContact = new HashMap<>();
                    for (int i = 0; i < n_subjects; i++) {
                        probabilityOfNotBeingInfectedDueToPreviousContact.put(String.valueOf(i + 1), new double[time_horizon]);
                        Arrays.fill(probabilityOfNotBeingInfectedDueToPreviousContact.get(String.valueOf(i + 1)), 1.0);
                    }
                    tracks_record.put(current_iteration, new Tracks(names, time_horizon));
                    for (Event event : events) {
                        int eventTime = Math.round(event.time / time_step); // this is the event time scaled by the time step
                        if (eventTime >= time_horizon) {
                            continue;
                        }
                        String[] involvedSubjects = event.involvedSubjects;

                        if (current_iteration == 0 && event.type.equals("External")) {
                            assert involvedSubjects.length == 1;
                            String involvedSubject = involvedSubjects[0];
                            double riskFactor_external = event.riskFactor;

                            double Pw_h_given_e_s_k_is_effective = 1.0;
                            double prior = 1.0;

                            for (Event entry : symptomsAndTests) {
                                if (!entry.involvedSubjects[0].equals(involvedSubject) || entry.time < event.time) {
                                    continue;
                                }

                                double desiredKey = Math.round((entry.time - event.time) / time_step);
                                if (desiredKey < 0.0 || !phi.containsKey(desiredKey)) {
                                    continue;
                                }

                                if (entry.type.equals("Symptoms") && entry.result.equals(true)) {
                                    Pw_h_given_e_s_k_is_effective *= phi.get(desiredKey);
                                    prior *= priorsValues[0];
                                } else if (entry.type.equals("Test")) {
                                    if (entry.result) {
                                        Pw_h_given_e_s_k_is_effective *= theta.get(desiredKey);
                                        prior *= priorsValues[1];
                                    } else {
                                        Pw_h_given_e_s_k_is_effective *= (1.0 - theta.get(desiredKey));
                                        prior *= priorsValues[2];
                                    }
                                }
                            }
                            double r_ext = Pw_h_given_e_s_k_is_effective * riskFactor_external / prior;

                            int offset_external = 0;
                            while (eventTime + offset_external < time_horizon) {
                                double previousValue = tracks_record.get(current_iteration).getSample(involvedSubject, eventTime + offset_external);

                                // Use the pre-calculated r_ext here
                                double newValue = stpnSolutionMap.get(offset_external) * r_ext * probabilityOfNotBeingInfectedDueToPreviousContact.get(involvedSubject)[eventTime + offset_external] + previousValue;

                                probabilityOfNotBeingInfectedDueToPreviousContact.get(involvedSubject)[eventTime + offset_external] *= (1.0 - r_ext);
                                tracks_record.get(current_iteration).editTrack(involvedSubject, eventTime + offset_external, newValue);
                                offset_external++;
                            }
                        } else if (current_iteration > 0 && event.type.equals("Internal")) {
                            String highestRiskSubject = null;
                            double highestRisk = 0.0;
                            String secondHighestRiskSubject = null;
                            double secondHighestRisk = 0.0;
                            for (String subject : involvedSubjects) { // get the highest risk subject
                                if (tracks_record.get(current_iteration - 1).getSample(subject, eventTime) > highestRisk) {
                                    highestRisk = tracks_record.get(current_iteration - 1).getSample(subject, eventTime);
                                    highestRiskSubject = subject;
                                }
                            }
                            for (String subject : involvedSubjects) { // get the second highest risk subject. This is used to update ~P for the highest risk subject
                                if (tracks_record.get(current_iteration - 1).getSample(subject, eventTime) > secondHighestRisk && tracks_record.get(current_iteration - 1).getSample(subject, eventTime) < highestRisk) {
                                    secondHighestRisk = tracks_record.get(current_iteration - 1).getSample(subject, eventTime);
                                    secondHighestRiskSubject = subject;
                                }
                            }
                            double riskFactor_internal = event.riskFactor;

                            HashMap<String, Double> r_int_map = new HashMap<>();
                            for (String subject : involvedSubjects) {
                                double Pw_h_given_e_s_k_is_effective = 1.0;
                                double prior = 1.0;

                                for (Event entry : symptomsAndTests) {
                                    if (!entry.involvedSubjects[0].equals(subject) || entry.time < event.time) {
                                        continue;
                                    }
                                    double desiredKey = Math.round((entry.time - event.time) / time_step);
                                    if (desiredKey < 0.0 || !phi.containsKey(desiredKey)) {
                                        continue;
                                    }

                                    if (entry.type.equals("Symptoms") && entry.result.equals(true)) {
                                        Pw_h_given_e_s_k_is_effective *= phi.get(desiredKey);
                                        prior *= priorsValues[0];
                                    } else if (entry.type.equals("Test")) {
                                        if (entry.result) {
                                            Pw_h_given_e_s_k_is_effective *= theta.get(desiredKey);
                                            prior *= priorsValues[1];
                                        } else {
                                            Pw_h_given_e_s_k_is_effective *= (1.0 - theta.get(desiredKey));
                                            prior *= priorsValues[2];
                                        }
                                    }
                                }
                                r_int_map.put(subject, Pw_h_given_e_s_k_is_effective * riskFactor_internal / prior);
                            }

                            int offset_internal = 0;
                            while (eventTime + offset_internal < time_horizon) {
                                for (String subject : involvedSubjects) {
                                    double previousValue = tracks_record.get(current_iteration).getSample(subject, eventTime + offset_internal);
                                    double q = subject.equals(highestRiskSubject) ? secondHighestRisk : highestRisk;

                                    // Retrieve the pre-calculated r_int
                                    double r_int = r_int_map.get(subject);

                                    double solutionValue = stpnSolutionMap.getOrDefault(offset_internal, 0.0);
                                    double newValue = r_int * q * solutionValue * probabilityOfNotBeingInfectedDueToPreviousContact.get(subject)[eventTime + offset_internal] + previousValue;

                                    probabilityOfNotBeingInfectedDueToPreviousContact.get(subject)[eventTime + offset_internal] *= (1.0 - q);
                                    tracks_record.get(current_iteration).editTrack(subject, eventTime + offset_internal, newValue);
                                }
                                offset_internal++;
                            }

                        }
                    }
                }


                Tracks tracks = new Tracks(names, time_horizon);
                for (int i = 0; i < n_subjects; i++) {
                    for (int j = 0; j < time_horizon; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n_iterations; k++) {
                            sum += tracks_record.get(k).getSample(String.valueOf(i + 1), j);
                        }
                        tracks.editTrack(String.valueOf(i + 1), j, sum);
                    }
                    Path jsonPath = Path.of(filePath);
                    Path parent = jsonPath.getParent();
                    String outName = documentName + "_" + priorsValues[0] + "," + priorsValues[1] + "," + priorsValues[2] + "_tracks_it" + n_iterations + ".json";
                    Path outPath = (parent != null) ? parent.resolve(outName) : Path.of(outName);
                    try (FileWriter file = new FileWriter(outPath.toFile())) {
                        file.write(tracks.toJson().toString());
                        file.flush();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                }
            }
        }
    }
}