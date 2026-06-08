
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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;


public class STPNAnalysis {
    private static final int OBSERVATION_CURVE_CACHE_VERSION = 1;
    private static final int OBSERVATION_CURVE_TIME_LIMIT_DAYS = 60;
    private static final double OBSERVATION_CURVE_BASE_DT_DAYS = 0.05;
    private static final String DEFAULT_ANALYSIS_PARAMETER_CASE_ID =
            "inf_mid__heal_mid__sym_mid__iso_mid__onset_mid__notif_mid__symdur_mid";
    private static final double TEST_GAMMA_SHAPE = 7.85;
    private static final double TEST_GAMMA_SCALE = 2.14;
    private static final double TEST_GAMMA_SCALING = 0.94;

    private static final class ErlangExponentialParams {
        final int erlangStages;
        final double erlangRate;
        final double exponentialRate;

        ErlangExponentialParams(int erlangStages, double erlangRate, double exponentialRate) {
            this.erlangStages = erlangStages;
            this.erlangRate = erlangRate;
            this.exponentialRate = exponentialRate;
        }

        String signature() {
            return String.format(
                    Locale.US,
                    "n=%d,l1=%.12f,l2=%.12f",
                    erlangStages,
                    erlangRate,
                    exponentialRate
            );
        }
    }

    private static final class CurveScenario {
        final String name;
        final double symptomaticProbability;
        final ErlangExponentialParams onset;
        final ErlangExponentialParams symptomDuration;
        final double psiP1;
        final double psiP2;
        final double psiL1;
        final double psiL2;

        CurveScenario(
                String name,
                double symptomaticProbability,
                ErlangExponentialParams onset,
                ErlangExponentialParams symptomDuration,
                double psiP1,
                double psiP2,
                double psiL1,
                double psiL2
        ) {
            this.name = name;
            this.symptomaticProbability = symptomaticProbability;
            this.onset = onset;
            this.symptomDuration = symptomDuration;
            this.psiP1 = psiP1;
            this.psiP2 = psiP2;
            this.psiL1 = psiL1;
            this.psiL2 = psiL2;
        }

        String signature() {
            return String.format(
                    Locale.US,
                    "scenario=%s;s=%.12f;onset={%s};duration={%s};psi=%.12f,%.12f,%.12f,%.12f;gamma=%.12f,%.12f,%.12f;dt=%.12f",
                    name,
                    symptomaticProbability,
                    onset.signature(),
                    symptomDuration.signature(),
                    psiP1,
                    psiP2,
                    psiL1,
                    psiL2,
                    TEST_GAMMA_SHAPE,
                    TEST_GAMMA_SCALE,
                    TEST_GAMMA_SCALING,
                    OBSERVATION_CURVE_BASE_DT_DAYS
            );
        }
    }

    private static final class GeneralizedErlangSpec {
        final String unitMeasure;
        final int erlangStages;
        final double erlangRate;
        final double exponentialRate;

        GeneralizedErlangSpec(String unitMeasure, int erlangStages, double erlangRate, double exponentialRate) {
            if (erlangStages <= 0) {
                throw new IllegalArgumentException("erlangStages must be greater than 0.");
            }
            this.unitMeasure = normalizeUnitMeasure(unitMeasure);
            this.erlangStages = erlangStages;
            this.erlangRate = erlangRate;
            this.exponentialRate = exponentialRate;
        }

        ErlangExponentialParams toPerHourParams() {
            return new ErlangExponentialParams(
                    erlangStages,
                    convertRate(erlangRate, unitMeasure, "hours"),
                    convertRate(exponentialRate, unitMeasure, "hours")
            );
        }

        ErlangExponentialParams toPerDayParams() {
            return new ErlangExponentialParams(
                    erlangStages,
                    convertRate(erlangRate, unitMeasure, "days"),
                    convertRate(exponentialRate, unitMeasure, "days")
            );
        }

        String signature() {
            return String.format(
                    Locale.US,
                    "unit=%s,n=%d,l1=%.12f,l2=%.12f",
                    unitMeasure,
                    erlangStages,
                    erlangRate,
                    exponentialRate
            );
        }
    }

    private static final class HyperExponentialParams {
        final double p1;
        final double p2;
        final double lambda1;
        final double lambda2;

        HyperExponentialParams(double p1, double p2, double lambda1, double lambda2) {
            this.p1 = p1;
            this.p2 = p2;
            this.lambda1 = lambda1;
            this.lambda2 = lambda2;
        }

        String signature() {
            return String.format(
                    Locale.US,
                    "p=%.12f,%.12f;l=%.12f,%.12f",
                    p1,
                    p2,
                    lambda1,
                    lambda2
            );
        }
    }

    private static final class HyperExponentialSpec {
        final String unitMeasure;
        final double p1;
        final double p2;
        final double lambda1;
        final double lambda2;

        HyperExponentialSpec(String unitMeasure, double p1, double p2, double lambda1, double lambda2) {
            this.unitMeasure = normalizeUnitMeasure(unitMeasure);
            this.p1 = p1;
            this.p2 = p2;
            this.lambda1 = lambda1;
            this.lambda2 = lambda2;
        }

        HyperExponentialParams toPerDayParams() {
            return new HyperExponentialParams(
                    p1,
                    p2,
                    convertRate(lambda1, unitMeasure, "days"),
                    convertRate(lambda2, unitMeasure, "days")
            );
        }

        String signature() {
            return String.format(
                    Locale.US,
                    "unit=%s,p=%.12f,%.12f;l=%.12f,%.12f",
                    unitMeasure,
                    p1,
                    p2,
                    lambda1,
                    lambda2
            );
        }
    }

    private static final class AnalysisParameters {
        final String caseId;
        final GeneralizedErlangSpec infectiousness;
        final GeneralizedErlangSpec healing;
        final double symptomaticProbability;
        final GeneralizedErlangSpec isolating;
        final GeneralizedErlangSpec symptomsOnset;
        final HyperExponentialSpec notificationToIsolation;
        final GeneralizedErlangSpec symptomaticPeriod;

        AnalysisParameters(
                String caseId,
                GeneralizedErlangSpec infectiousness,
                GeneralizedErlangSpec healing,
                double symptomaticProbability,
                GeneralizedErlangSpec isolating,
                GeneralizedErlangSpec symptomsOnset,
                HyperExponentialSpec notificationToIsolation,
                GeneralizedErlangSpec symptomaticPeriod
        ) {
            if (symptomaticProbability < 0.0 || symptomaticProbability > 1.0) {
                throw new IllegalArgumentException("symptomaticProbability must be between 0 and 1.");
            }
            if (healing.erlangStages != 1) {
                throw new IllegalArgumentException(
                        "The Java STPN model expects the healing transition to use exactly one Erlang stage."
                );
            }
            this.caseId = caseId;
            this.infectiousness = infectiousness;
            this.healing = healing;
            this.symptomaticProbability = symptomaticProbability;
            this.isolating = isolating;
            this.symptomsOnset = symptomsOnset;
            this.notificationToIsolation = notificationToIsolation;
            this.symptomaticPeriod = symptomaticPeriod;
        }

        CurveScenario curveScenario() {
            HyperExponentialParams notificationPerDay = notificationToIsolation.toPerDayParams();
            return new CurveScenario(
                    caseId,
                    symptomaticProbability,
                    symptomsOnset.toPerDayParams(),
                    symptomaticPeriod.toPerDayParams(),
                    notificationPerDay.p1,
                    notificationPerDay.p2,
                    notificationPerDay.lambda1,
                    notificationPerDay.lambda2
            );
        }

        String signature() {
            return String.format(
                    Locale.US,
                    "case=%s;infectiousness={%s};healing={%s};symptoms=%.12f;isolating={%s};onset={%s};notification={%s};symptomatic_period={%s}",
                    caseId,
                    infectiousness.signature(),
                    healing.signature(),
                    symptomaticProbability,
                    isolating.signature(),
                    symptomsOnset.signature(),
                    notificationToIsolation.signature(),
                    symptomaticPeriod.signature()
            );
        }
    }

    private static final class ObservationCurves {
        final double[] phi;
        final double[] theta;
        final double[] psiSurvival;

        ObservationCurves(double[] phi, double[] theta, double[] psiSurvival) {
            this.phi = phi;
            this.theta = theta;
            this.psiSurvival = psiSurvival;
        }

        int length() {
            return phi.length;
        }
    }

    @SuppressWarnings("unchecked")
    public static <R, S> TransientSolution<R, S> buildModel(
            int samples,
            float step,
            AnalysisParameters analysisParameters
    ) {
        ErlangExponentialParams isolating = analysisParameters.isolating.toPerHourParams();
        ErlangExponentialParams healing = analysisParameters.healing.toPerHourParams();
        ErlangExponentialParams infectiousness = analysisParameters.infectiousness.toPerHourParams();
        ErlangExponentialParams symptomsOnset = analysisParameters.symptomsOnset.toPerHourParams();
        double symptomaticProbability = analysisParameters.symptomaticProbability;
        double asymptomaticProbability = clamp01(1.0 - symptomaticProbability);

        PetriNet net = new PetriNet();
        Marking marking = new Marking();

        // Places and transitions
        Place Asymptomatic = net.addPlace("Asymptomatic");
        Place DevelopingSymptoms = net.addPlace("DevelopingSymptoms");
        Place EffectiveContact = net.addPlace("EffectiveContact");
        Place Healed = net.addPlace("Healed");
        Place Infected = net.addPlace("Infected");
        Place Infectious = net.addPlace("Infectious");
        Place Isolated = net.addPlace("Isolated");
        Place Symptomatic = net.addPlace("Symptomatic");
        Place Symptomatology = net.addPlace("Symptomatology");
        Place _healing = net.addPlace("_healing");
        Place _infectiousness = net.addPlace("_infectiousness");
        Place _isolating = net.addPlace("_isolating");
        Place _symptomsonset = net.addPlace("_symptomsonset");
        Transition Isolating_erlang = net.addTransition("Isolating_erlang");
        Transition Isolating_exp = net.addTransition("Isolating_exp");
        Transition effectiveContact = net.addTransition("effectiveContact");
        Transition healing_exp1 = net.addTransition("healing_exp1");
        Transition healing_exp2 = net.addTransition("healing_exp2");
        Transition infectiousness_erl = net.addTransition("infectiousness_erl");
        Transition infectiousness_exp = net.addTransition("infectiousness_exp");
        Transition noSymptoms = net.addTransition("noSymptoms");
        Transition symptoms = net.addTransition("symptoms");
        Transition symptomsonset_erl = net.addTransition("symptomsonset_erl");
        Transition symptomsonset_exp = net.addTransition("symptomsonset_exp");

        // Petri net arcs
        net.addPrecondition(Infectious, healing_exp1);
        net.addPostcondition(symptomsonset_erl, _symptomsonset);
        net.addPostcondition(infectiousness_exp, Infectious);
        net.addPrecondition(Symptomatology, symptoms);
        net.addPostcondition(healing_exp2, Healed);
        net.addPrecondition(Symptomatic, Isolating_erlang);
        net.addPrecondition(_healing, healing_exp2);
        net.addPostcondition(noSymptoms, Asymptomatic);
        net.addPrecondition(Symptomatology, noSymptoms);
        net.addPostcondition(symptoms, DevelopingSymptoms);
        net.addPostcondition(effectiveContact, Infected);
        net.addPrecondition(Infected, infectiousness_erl);
        net.addPostcondition(Isolating_erlang, _isolating);
        net.addPostcondition(effectiveContact, Symptomatology);
        net.addPrecondition(Asymptomatic, healing_exp1);
        net.addPrecondition(_infectiousness, infectiousness_exp);
        net.addPrecondition(_isolating, Isolating_exp);
        net.addPrecondition(DevelopingSymptoms, symptomsonset_erl);
        net.addPostcondition(symptomsonset_exp, Symptomatic);
        net.addPrecondition(EffectiveContact, effectiveContact);
        net.addPostcondition(healing_exp1, _healing);
        net.addPostcondition(Isolating_exp, Isolated);
        net.addPrecondition(_symptomsonset, symptomsonset_exp);
        net.addPostcondition(infectiousness_erl, _infectiousness);

        // Initial marking and transition features
        marking.setTokens(Asymptomatic, 0);
        marking.setTokens(DevelopingSymptoms, 0);
        marking.setTokens(EffectiveContact, 1);
        marking.setTokens(Healed, 0);
        marking.setTokens(Infected, 0);
        marking.setTokens(Infectious, 0);
        marking.setTokens(Isolated, 0);
        marking.setTokens(Symptomatic, 0);
        marking.setTokens(Symptomatology, 0);
        marking.setTokens(_healing, 0);
        marking.setTokens(_infectiousness, 0);
        marking.setTokens(_isolating, 0);
        marking.setTokens(_symptomsonset, 0);
        Isolating_erlang.addFeature(StochasticTransitionFeature.newErlangInstance(
                isolating.erlangStages,
                new BigDecimal(Double.toString(isolating.erlangRate))
        ));
        Isolating_exp.addFeature(StochasticTransitionFeature.newExponentialInstance(
                new BigDecimal("1"),
                MarkingExpr.from(Double.toString(isolating.exponentialRate), net)
        ));
        effectiveContact.addFeature(StochasticTransitionFeature.newDeterministicInstance(new BigDecimal("0"), MarkingExpr.from("1", net)));
        effectiveContact.addFeature(new Priority(0));
        healing_exp1.addFeature(StochasticTransitionFeature.newExponentialInstance(
                new BigDecimal("1"),
                MarkingExpr.from(Double.toString(healing.exponentialRate), net)
        ));
        healing_exp2.addFeature(StochasticTransitionFeature.newExponentialInstance(
                new BigDecimal("1"),
                MarkingExpr.from(Double.toString(healing.erlangRate), net)
        ));
        infectiousness_erl.addFeature(StochasticTransitionFeature.newErlangInstance(
                infectiousness.erlangStages,
                new BigDecimal(Double.toString(infectiousness.erlangRate))
        ));
        infectiousness_exp.addFeature(StochasticTransitionFeature.newExponentialInstance(
                new BigDecimal("1"),
                MarkingExpr.from(Double.toString(infectiousness.exponentialRate), net)
        ));
        noSymptoms.addFeature(StochasticTransitionFeature.newDeterministicInstance(
                new BigDecimal("0"),
                MarkingExpr.from(Double.toString(asymptomaticProbability), net)
        ));
        noSymptoms.addFeature(new Priority(0));
        symptoms.addFeature(StochasticTransitionFeature.newDeterministicInstance(
                new BigDecimal("0"),
                MarkingExpr.from(Double.toString(symptomaticProbability), net)
        ));
        symptoms.addFeature(new Priority(0));
        symptomsonset_erl.addFeature(StochasticTransitionFeature.newErlangInstance(
                symptomsOnset.erlangStages,
                new BigDecimal(Double.toString(symptomsOnset.erlangRate))
        ));
        symptomsonset_exp.addFeature(StochasticTransitionFeature.newExponentialInstance(
                new BigDecimal("1"),
                MarkingExpr.from(Double.toString(symptomsOnset.exponentialRate), net)
        ));

        // Run analysis
        TreeTransient analysis = TreeTransient.builder()
                .greedyPolicy(new BigDecimal(samples), BigDecimal.ZERO)
                .timeStep(new BigDecimal(step))
                .build();

        TransientSolution<Marking, Marking> result = analysis.compute(net, marking);

        var rewardRates = TransientSolution.rewardRates("Infectious==1 && Isolated==0");
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

    private static double clamp01(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return 0.0;
        }
        return Math.max(0.0, Math.min(1.0, value));
    }

    private static AnalysisParameters defaultAnalysisParameters() {
        return new AnalysisParameters(
                DEFAULT_ANALYSIS_PARAMETER_CASE_ID,
                new GeneralizedErlangSpec("hours", 2, 0.060879, 0.02051375),
                new GeneralizedErlangSpec("hours", 1, 0.01115625, 0.0055816667),
                0.649,
                new GeneralizedErlangSpec("hours", 3, 0.0336325, 0.016447083),
                new GeneralizedErlangSpec("hours", 2, 0.02255375, 0.01341875),
                new HyperExponentialSpec("days", 0.88188, 0.11812, 4.64146, 0.62170),
                new GeneralizedErlangSpec("days", 2, 0.60118, 0.21962)
        );
    }

    private static String normalizeUnitMeasure(String unitMeasure) {
        String normalized = unitMeasure == null ? "" : unitMeasure.trim().toLowerCase(Locale.ROOT);
        if (normalized.equals("hour") || normalized.equals("hours")) {
            return "hours";
        }
        if (normalized.equals("day") || normalized.equals("days")) {
            return "days";
        }
        throw new IllegalArgumentException("Unsupported unit measure: " + unitMeasure);
    }

    private static double convertRate(double rate, String unitMeasure, String targetUnitMeasure) {
        String source = normalizeUnitMeasure(unitMeasure);
        String target = normalizeUnitMeasure(targetUnitMeasure);
        if (source.equals(target)) {
            return rate;
        }
        if (source.equals("hours") && target.equals("days")) {
            return rate * 24.0;
        }
        if (source.equals("days") && target.equals("hours")) {
            return rate / 24.0;
        }
        throw new IllegalArgumentException(
                "Cannot convert rate from " + unitMeasure + " to " + targetUnitMeasure + "."
        );
    }

    private static JsonObject requireJsonObject(JsonObject jsonObject, String key) {
        if (jsonObject == null || !jsonObject.has(key) || !jsonObject.get(key).isJsonObject()) {
            throw new IllegalArgumentException("Missing JSON object field '" + key + "'.");
        }
        return jsonObject.getAsJsonObject(key);
    }

    private static String requireString(JsonObject jsonObject, String key) {
        if (jsonObject == null || !jsonObject.has(key) || jsonObject.get(key).isJsonNull()) {
            throw new IllegalArgumentException("Missing string field '" + key + "'.");
        }
        return jsonObject.get(key).getAsString();
    }

    private static String optionalString(JsonObject jsonObject, String key, String fallback) {
        if (jsonObject == null || !jsonObject.has(key) || jsonObject.get(key).isJsonNull()) {
            return fallback;
        }
        return jsonObject.get(key).getAsString();
    }

    private static int requireInt(JsonObject jsonObject, String key) {
        if (jsonObject == null || !jsonObject.has(key) || jsonObject.get(key).isJsonNull()) {
            throw new IllegalArgumentException("Missing integer field '" + key + "'.");
        }
        return jsonObject.get(key).getAsInt();
    }

    private static double requireDouble(JsonObject jsonObject, String key) {
        if (jsonObject == null || !jsonObject.has(key) || jsonObject.get(key).isJsonNull()) {
            throw new IllegalArgumentException("Missing numeric field '" + key + "'.");
        }
        return jsonObject.get(key).getAsDouble();
    }

    private static GeneralizedErlangSpec parseGeneralizedErlangSpec(JsonObject transitions, String key) {
        JsonObject transition = requireJsonObject(transitions, key);
        return new GeneralizedErlangSpec(
                requireString(transition, "unit_measure"),
                requireInt(transition, "erlang_stages"),
                requireDouble(transition, "erlang_lambda"),
                requireDouble(transition, "exponential_lambda")
        );
    }

    private static HyperExponentialSpec parseHyperExponentialSpec(JsonObject transitions, String key) {
        JsonObject transition = requireJsonObject(transitions, key);
        String distribution = optionalString(transition, "distribution", "hyperexponential");
        if (!distribution.equalsIgnoreCase("hyperexponential")) {
            throw new IllegalArgumentException(
                    "Unsupported notification-to-isolation distribution: " + distribution
            );
        }
        return new HyperExponentialSpec(
                requireString(transition, "unit_measure"),
                requireDouble(transition, "p1"),
                requireDouble(transition, "p2"),
                requireDouble(transition, "lambda1"),
                requireDouble(transition, "lambda2")
        );
    }

    private static AnalysisParameters loadAnalysisParameters(String parameterBundlePath) {
        if (parameterBundlePath == null || parameterBundlePath.isBlank()) {
            return defaultAnalysisParameters();
        }

        JsonObject bundle = JsonFileReader.readJsonFromFile(parameterBundlePath);
        if (bundle == null) {
            throw new IllegalArgumentException("Failed to read parameter bundle: " + parameterBundlePath);
        }

        JsonObject transitions = requireJsonObject(bundle, "transitions");
        return new AnalysisParameters(
                optionalString(bundle, "case_id", "custom_parameter_bundle"),
                parseGeneralizedErlangSpec(transitions, "infectiousness"),
                parseGeneralizedErlangSpec(transitions, "healing"),
                requireDouble(requireJsonObject(transitions, "symptoms"), "true"),
                parseGeneralizedErlangSpec(transitions, "isolating"),
                parseGeneralizedErlangSpec(transitions, "symptomsOnset"),
                parseHyperExponentialSpec(transitions, "notificationToIsolation"),
                parseGeneralizedErlangSpec(transitions, "symptomaticPeriod")
        );
    }

    private static boolean hasObservationCurveOffset(ObservationCurves curves, int offset) {
        return offset >= 0 && offset < curves.length();
    }

    private static double computeKernelValue(
            HashMap<Integer, Double> stpnSolutionMap,
            int offset,
            Integer firstPositiveTestOffset,
            double[] psiSurvival
    ) {
        if (firstPositiveTestOffset == null || offset < firstPositiveTestOffset) {
            return stpnSolutionMap.getOrDefault(offset, 0.0);
        }

        int elapsedOffsetSincePositiveTest = offset - firstPositiveTestOffset;
        if (elapsedOffsetSincePositiveTest >= psiSurvival.length) {
            return 0.0;
        }
        return psiSurvival[elapsedOffsetSincePositiveTest];
    }

    private static ObservationCurves loadOrCreateObservationCurves(
            AnalysisParameters analysisParameters,
            float timeStepHours
    ) throws IOException {
        CurveScenario scenario = analysisParameters.curveScenario();
        int horizonSteps = observationCurveHorizonSteps(timeStepHours);
        Path cachePath = observationCurveCachePath(scenario, timeStepHours);
        ObservationCurves curves = loadObservationCurves(cachePath, scenario, timeStepHours, horizonSteps);
        if (curves != null) {
            System.out.println("Loaded observation curves from " + cachePath);
            return curves;
        }

        System.out.println("Observation curve cache not found or stale. Computing " + scenario.name + " curves...");
        curves = computeObservationCurves(scenario, timeStepHours, horizonSteps);
        writeObservationCurves(cachePath, scenario, timeStepHours, curves);
        System.out.println("Observation curves saved to " + cachePath);
        return curves;
    }

    private static int observationCurveHorizonSteps(float timeStepHours) {
        return (int) Math.ceil(OBSERVATION_CURVE_TIME_LIMIT_DAYS * 24.0 / (double) timeStepHours);
    }

    private static String timeStepLabel(float timeStepHours) {
        String label = String.format(Locale.US, "%.6f", (double) timeStepHours);
        while (label.contains(".") && label.endsWith("0")) {
            label = label.substring(0, label.length() - 1);
        }
        if (label.endsWith(".")) {
            label = label.substring(0, label.length() - 1);
        }
        return label.replace("-", "m").replace(".", "p");
    }

    private static Path observationCurveCachePath(CurveScenario scenario, float timeStepHours) {
        String scenarioLabel = scenario.name.replaceAll("[^A-Za-z0-9._-]+", "_");
        return Path.of("observation_curves_" + scenarioLabel + "_step" + timeStepLabel(timeStepHours) + ".csv");
    }

    private static ObservationCurves loadObservationCurves(
            Path cachePath,
            CurveScenario scenario,
            float timeStepHours,
            int horizonSteps
    ) {
        if (!Files.exists(cachePath)) {
            return null;
        }

        HashMap<String, String> metadata = new HashMap<>();
        double[] phi = new double[horizonSteps];
        double[] theta = new double[horizonSteps];
        double[] psiSurvival = new double[horizonSteps];
        int loadedRows = 0;
        boolean foundHeader = false;

        try (BufferedReader reader = Files.newBufferedReader(cachePath)) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                if (line.startsWith("#")) {
                    String payload = line.substring(1).trim();
                    int separator = payload.indexOf('=');
                    if (separator > 0) {
                        metadata.put(payload.substring(0, separator).trim(), payload.substring(separator + 1).trim());
                    }
                    continue;
                }
                if (!foundHeader) {
                    if (!line.equals("index,phi,theta,psi_survival")) {
                        return null;
                    }
                    foundHeader = true;
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length != 4) {
                    return null;
                }
                int index = Integer.parseInt(parts[0]);
                if (index < 0 || index >= horizonSteps) {
                    return null;
                }
                phi[index] = Double.parseDouble(parts[1]);
                theta[index] = Double.parseDouble(parts[2]);
                psiSurvival[index] = Double.parseDouble(parts[3]);
                loadedRows++;
            }
        } catch (IOException | NumberFormatException e) {
            return null;
        }

        if (!foundHeader || loadedRows != horizonSteps) {
            return null;
        }
        if (!String.valueOf(OBSERVATION_CURVE_CACHE_VERSION).equals(metadata.get("version"))) {
            return null;
        }
        if (!scenario.name.equals(metadata.get("scenario"))) {
            return null;
        }
        if (!scenario.signature().equals(metadata.get("parameter_signature"))) {
            return null;
        }
        if (!String.valueOf(horizonSteps).equals(metadata.get("horizon_steps"))) {
            return null;
        }
        try {
            double cachedTimeStepHours = Double.parseDouble(metadata.getOrDefault("time_step_hours", "NaN"));
            if (Math.abs(cachedTimeStepHours - (double) timeStepHours) > 1e-9) {
                return null;
            }
        } catch (NumberFormatException e) {
            return null;
        }
        return new ObservationCurves(phi, theta, psiSurvival);
    }

    private static void writeObservationCurves(
            Path cachePath,
            CurveScenario scenario,
            float timeStepHours,
            ObservationCurves curves
    ) throws IOException {
        try (BufferedWriter writer = Files.newBufferedWriter(cachePath)) {
            writer.write("# version=" + OBSERVATION_CURVE_CACHE_VERSION);
            writer.newLine();
            writer.write("# scenario=" + scenario.name);
            writer.newLine();
            writer.write("# parameter_signature=" + scenario.signature());
            writer.newLine();
            writer.write(String.format(Locale.US, "# time_step_hours=%.12f%n", (double) timeStepHours));
            writer.write("# horizon_steps=" + curves.length());
            writer.newLine();
            writer.write("# curve_time_limit_days=" + OBSERVATION_CURVE_TIME_LIMIT_DAYS);
            writer.newLine();
            writer.write(String.format(Locale.US, "# base_dt_days=%.12f%n", OBSERVATION_CURVE_BASE_DT_DAYS));
            writer.write("index,phi,theta,psi_survival");
            writer.newLine();

            for (int i = 0; i < curves.length(); i++) {
                writer.write(String.format(
                        Locale.US,
                        "%d,%.17g,%.17g,%.17g%n",
                        i,
                        curves.phi[i],
                        curves.theta[i],
                        curves.psiSurvival[i]
                ));
            }
        }
    }

    private static ObservationCurves computeObservationCurves(
            CurveScenario scenario,
            float timeStepHours,
            int horizonSteps
    ) {
        double[] phiBase = computePhiBaseCurve(scenario);
        double[] thetaBase = computeThetaBaseCurve(scenario);
        double[] phi = new double[horizonSteps];
        double[] theta = new double[horizonSteps];
        double[] psiSurvival = new double[horizonSteps];

        for (int i = 0; i < horizonSteps; i++) {
            double tDays = i * (double) timeStepHours / 24.0;
            phi[i] = clamp01(sampleBaseCurve(phiBase, tDays));
            theta[i] = clamp01(sampleBaseCurve(thetaBase, tDays));
            psiSurvival[i] = clamp01(
                    scenario.psiP1 * Math.exp(-scenario.psiL1 * tDays)
                            + scenario.psiP2 * Math.exp(-scenario.psiL2 * tDays)
            );
        }

        return new ObservationCurves(phi, theta, psiSurvival);
    }

    private static double[] computePhiBaseCurve(CurveScenario scenario) {
        int baseSamples = baseCurveSamples();
        double[] onsetPdf = generalizedErlangPdf(scenario.onset, baseSamples);
        double[] durationPdf = generalizedErlangPdf(scenario.symptomDuration, baseSamples);
        double[] durationSurvival = new double[baseSamples];
        double durationCdf = 0.0;

        for (int i = 0; i < baseSamples; i++) {
            durationCdf += durationPdf[i] * OBSERVATION_CURVE_BASE_DT_DAYS;
            durationSurvival[i] = clamp01(1.0 - durationCdf);
        }

        double[] phiConv = convolve(onsetPdf, durationSurvival, baseSamples, OBSERVATION_CURVE_BASE_DT_DAYS);
        for (int i = 0; i < baseSamples; i++) {
            phiConv[i] = clamp01(scenario.symptomaticProbability * phiConv[i]);
        }
        return phiConv;
    }

    private static double[] computeThetaBaseCurve(CurveScenario scenario) {
        int baseSamples = baseCurveSamples();
        double[] onsetPdf = generalizedErlangPdf(scenario.onset, baseSamples);
        double[] positiveSinceSymptomOnset = new double[baseSamples];

        for (int i = 0; i < baseSamples; i++) {
            double tDays = i * OBSERVATION_CURVE_BASE_DT_DAYS;
            positiveSinceSymptomOnset[i] = clamp01(
                    TEST_GAMMA_SCALING * gammaSurvival(TEST_GAMMA_SHAPE, TEST_GAMMA_SCALE, tDays)
            );
        }

        double[] thetaConv = convolve(onsetPdf, positiveSinceSymptomOnset, baseSamples, OBSERVATION_CURVE_BASE_DT_DAYS);
        for (int i = 0; i < baseSamples; i++) {
            thetaConv[i] = clamp01(thetaConv[i]);
        }
        return thetaConv;
    }

    private static int baseCurveSamples() {
        return (int) Math.ceil(OBSERVATION_CURVE_TIME_LIMIT_DAYS / OBSERVATION_CURVE_BASE_DT_DAYS);
    }

    private static double[] generalizedErlangPdf(ErlangExponentialParams params, int samples) {
        double[] erlangPdf = new double[samples];
        double[] exponentialPdf = new double[samples];
        for (int i = 0; i < samples; i++) {
            double tDays = i * OBSERVATION_CURVE_BASE_DT_DAYS;
            erlangPdf[i] = erlangPdf(tDays, params.erlangStages, params.erlangRate);
            exponentialPdf[i] = exponentialPdf(tDays, params.exponentialRate);
        }
        return convolve(erlangPdf, exponentialPdf, samples, OBSERVATION_CURVE_BASE_DT_DAYS);
    }

    private static double erlangPdf(double t, int stages, double rate) {
        if (t < 0.0 || stages <= 0 || rate <= 0.0) {
            return 0.0;
        }
        if (t == 0.0) {
            return stages == 1 ? rate : 0.0;
        }
        double logPdf = stages * Math.log(rate) + (stages - 1) * Math.log(t) - rate * t - logFactorial(stages - 1);
        return Math.exp(logPdf);
    }

    private static double exponentialPdf(double t, double rate) {
        if (t < 0.0 || rate <= 0.0) {
            return 0.0;
        }
        return rate * Math.exp(-rate * t);
    }

    private static double logFactorial(int n) {
        double result = 0.0;
        for (int i = 2; i <= n; i++) {
            result += Math.log(i);
        }
        return result;
    }

    private static double[] convolve(double[] a, double[] b, int length, double dt) {
        double[] result = new double[length];
        for (int i = 0; i < length; i++) {
            double sum = 0.0;
            for (int j = 0; j <= i; j++) {
                sum += a[j] * b[i - j];
            }
            result[i] = sum * dt;
        }
        return result;
    }

    private static double sampleBaseCurve(double[] curve, double tDays) {
        if (tDays < 0.0) {
            return 0.0;
        }
        double position = tDays / OBSERVATION_CURVE_BASE_DT_DAYS;
        int lowerIndex = (int) Math.floor(position);
        if (lowerIndex < 0) {
            return 0.0;
        }
        if (lowerIndex >= curve.length - 1) {
            return lowerIndex == curve.length - 1 ? curve[lowerIndex] : 0.0;
        }
        double fraction = position - lowerIndex;
        return curve[lowerIndex] * (1.0 - fraction) + curve[lowerIndex + 1] * fraction;
    }

    private static double gammaSurvival(double shape, double scale, double t) {
        if (t <= 0.0) {
            return 1.0;
        }
        return regularizedGammaQ(shape, t / scale);
    }

    private static double regularizedGammaQ(double a, double x) {
        if (a <= 0.0 || x < 0.0) {
            return Double.NaN;
        }
        if (x == 0.0) {
            return 1.0;
        }
        if (x < a + 1.0) {
            return 1.0 - regularizedGammaPSeries(a, x);
        }
        return regularizedGammaQContinuedFraction(a, x);
    }

    private static double regularizedGammaPSeries(double a, double x) {
        final int maxIterations = 10000;
        final double epsilon = 1e-14;
        double sum = 1.0 / a;
        double term = sum;

        for (int n = 1; n <= maxIterations; n++) {
            term *= x / (a + n);
            sum += term;
            if (Math.abs(term) < Math.abs(sum) * epsilon) {
                double logTerm = -x + a * Math.log(x) - logGamma(a);
                return clamp01(sum * Math.exp(logTerm));
            }
        }
        double logTerm = -x + a * Math.log(x) - logGamma(a);
        return clamp01(sum * Math.exp(logTerm));
    }

    private static double regularizedGammaQContinuedFraction(double a, double x) {
        final int maxIterations = 10000;
        final double epsilon = 1e-14;
        final double fpMin = 1e-300;
        double b = x + 1.0 - a;
        double c = 1.0 / fpMin;
        double d = 1.0 / Math.max(b, fpMin);
        double h = d;

        for (int i = 1; i <= maxIterations; i++) {
            double an = -i * (i - a);
            b += 2.0;
            d = an * d + b;
            if (Math.abs(d) < fpMin) {
                d = fpMin;
            }
            c = b + an / c;
            if (Math.abs(c) < fpMin) {
                c = fpMin;
            }
            d = 1.0 / d;
            double delta = d * c;
            h *= delta;
            if (Math.abs(delta - 1.0) < epsilon) {
                break;
            }
        }

        double logTerm = -x + a * Math.log(x) - logGamma(a);
        return clamp01(Math.exp(logTerm) * h);
    }

    private static double logGamma(double x) {
        double[] coefficients = {
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7
        };

        if (x < 0.5) {
            return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * x)) - logGamma(1.0 - x);
        }

        x -= 1.0;
        double sum = 0.99999999999980993;
        for (int i = 0; i < coefficients.length; i++) {
            sum += coefficients[i] / (x + i + 1.0);
        }
        double t = x + coefficients.length - 0.5;
        return 0.5 * Math.log(2.0 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(sum);
    }

    private static HashMap<String, ArrayList<Event>> indexEventsBySubject(Collection<Event> events) {
        HashMap<String, ArrayList<Event>> eventsBySubject = new HashMap<>();
        for (Event event : events) {
            for (String subject : event.involvedSubjects) {
                eventsBySubject
                        .computeIfAbsent(subject, ignored -> new ArrayList<>())
                        .add(event);
            }
        }
        return eventsBySubject;
    }

    private static boolean loadStpnSolution(
            File stpnCsvFile,
            AnalysisParameters analysisParameters,
            float timeStepHours,
            HashMap<Integer, Double> stpnSolutionMap
    ) {
        HashMap<String, String> metadata = new HashMap<>();
        boolean foundHeader = false;

        try (BufferedReader reader = new BufferedReader(new FileReader(stpnCsvFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue;
                }
                if (line.startsWith("#")) {
                    String payload = line.substring(1).trim();
                    int separator = payload.indexOf('=');
                    if (separator > 0) {
                        metadata.put(payload.substring(0, separator).trim(), payload.substring(separator + 1).trim());
                    }
                    continue;
                }
                if (!foundHeader) {
                    if (!line.equals("Time,State,Value")) {
                        return false;
                    }
                    foundHeader = true;
                    continue;
                }

                String[] parts = line.split(",");
                if (parts.length != 3) {
                    return false;
                }
                int time = Integer.parseInt(parts[0]);
                int state = Integer.parseInt(parts[1]);
                double value = Double.parseDouble(parts[2]);
                if (state == 0) {
                    stpnSolutionMap.put(time, value);
                }
            }
        } catch (IOException | NumberFormatException e) {
            return false;
        }

        if (!foundHeader || stpnSolutionMap.isEmpty()) {
            return false;
        }
        if (!analysisParameters.signature().equals(metadata.get("parameter_signature"))) {
            return false;
        }
        try {
            double cachedTimeStepHours = Double.parseDouble(metadata.getOrDefault("time_step_hours", "NaN"));
            if (Math.abs(cachedTimeStepHours - (double) timeStepHours) > 1e-9) {
                return false;
            }
        } catch (NumberFormatException e) {
            return false;
        }
        return true;
    }

    private static void writeStpnSolution(
            File stpnCsvFile,
            TransientSolution<Integer, Double> solution,
            AnalysisParameters analysisParameters,
            float timeStepHours,
            HashMap<Integer, Double> stpnSolutionMap
    ) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(stpnCsvFile))) {
            writer.write("# case_id=" + analysisParameters.caseId);
            writer.newLine();
            writer.write("# parameter_signature=" + analysisParameters.signature());
            writer.newLine();
            writer.write(String.format(Locale.US, "# time_step_hours=%.12f%n", (double) timeStepHours));
            writer.write("Time,State,Value");
            writer.newLine();

            for (int i = 0; i < solution.getSolution().length; i++) {
                double value = solution.getSolution()[i][0][0];
                writer.write(i + ",0," + value);
                writer.newLine();
                stpnSolutionMap.put(i, value);
            }
        }
    }

    private static String getArgValue(String[] args, String name) {
        for (int i = 0; i < args.length - 1; i++) {
            if (args[i].equals(name)) {
                return args[i + 1];
            }
        }
        return null;
    }

    public static void main(String[] args) throws Exception {
        float time_step = 1.0f;
        String timeStepArg = getArgValue(args, "--time-step");
        if (timeStepArg != null) {
            time_step = Float.parseFloat(timeStepArg);
            if (time_step <= 0.0f) {
                throw new IllegalArgumentException("time_step must be greater than 0");
            }
        }
        String stpnSolutionPath = getArgValue(args, "--stpn-solution-path");
        if (stpnSolutionPath == null || stpnSolutionPath.isBlank()) {
            stpnSolutionPath = "stpn_solution.csv";
        }
        int requestedIterations = 3;
        String iterationsArg = getArgValue(args, "--iterations");
        if (iterationsArg != null) {
            requestedIterations = Integer.parseInt(iterationsArg);
            if (requestedIterations <= 0) {
                throw new IllegalArgumentException("iterations must be greater than 0");
            }
        }
        String parameterBundlePath = getArgValue(args, "--parameter-bundle");
        boolean precomputeOnly = Arrays.asList(args).contains("--precompute-only");
        AnalysisParameters analysisParameters = loadAnalysisParameters(parameterBundlePath);


        LinkedHashSet<String> discoveredJsonFiles = new LinkedHashSet<>();
        Path startPath = Path.of(".");
        // Recursively search this folder and subfolders for files ending with "simulated.json"
        try (java.util.stream.Stream<Path> walk = Files.walk(startPath)) {
            walk.filter(Files::isRegularFile)
                .filter(p -> p.toString().endsWith("simulated.json"))
                .map(Path::toString)
                .forEach(discoveredJsonFiles::add);
        } catch (IOException e) {
            e.printStackTrace();
        }

        List<String> jsonFiles = new ArrayList<>(discoveredJsonFiles);
        jsonFiles.forEach(System.out::println);
        System.out.println("Using analysis parameter bundle: " + analysisParameters.caseId);
        ObservationCurves observationCurves = loadOrCreateObservationCurves(analysisParameters, time_step);
        int time_limit = 84;
        int curve_time_limit = 63;

        HashMap<Integer, Double> stpnSolutionMap = new HashMap<>();
        File stpnCsvFile = new File(stpnSolutionPath);
        boolean loadedExistingSolution = stpnCsvFile.exists()
                && loadStpnSolution(stpnCsvFile, analysisParameters, time_step, stpnSolutionMap);
        if (!loadedExistingSolution) {
            System.out.println(stpnCsvFile.getName() + " not found or stale. Generating new solution...");
            int curveSamples = Math.round((curve_time_limit * 24.0f) / time_step);
            TransientSolution<Integer, Double> solution = buildModel(curveSamples, time_step, analysisParameters);

            try {
                stpnSolutionMap.clear();
                writeStpnSolution(stpnCsvFile, solution, analysisParameters, time_step, stpnSolutionMap);
                System.out.println("STPN solution saved to " + stpnCsvFile.getName());
            } catch (IOException e) {
                System.err.println("Error writing STPN solution to CSV file.");
                e.printStackTrace();
            }
        } else {
            System.out.println("Loaded existing solution from " + stpnCsvFile.getName() + ".");
            System.out.println("Successfully loaded " + stpnSolutionMap.size() + " time points.");
        }

        if (precomputeOnly) {
            return;
        }

        long coreAnalysisStartedAt = System.nanoTime();
        for (int repetition = 0; repetition < 1; repetition++) {
            for (int documentId = 0; documentId < jsonFiles.size(); documentId++) {
                int n_iterations = requestedIterations;
                double[] priorsValues = {0.5, 0.5, 0.5};
                String filePath = jsonFiles.get(documentId);
                String documentName = filePath.substring(filePath.lastIndexOf("\\") + 1, filePath.lastIndexOf("."));
                JsonObject jsonObject = JsonFileReader.readJsonFromFile(filePath);
                int n_subjects = 0;

                List<Event> events = new ArrayList<>();
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
                                event.get("time").getAsDouble(),
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
                Tracks[] tracksRecord = new Tracks[n_iterations];

                // Here we iterate over the symptoms and tests to gather that information
                ArrayList<Event> symptomsAndTests = new ArrayList<>();
                for (Event event : events) {
                    if (((event.type.equals("Symptoms") && Boolean.TRUE.equals(event.result)) || event.type.equals("Test")) && event.time < time_limit) { // we assume that the symptoms are only for the subject involved in the external contact
                        symptomsAndTests.add(event);
                    }
                }
                HashMap<String, ArrayList<Event>> symptomsAndTestsBySubject = indexEventsBySubject(symptomsAndTests);
                System.out.println("Time horizon: " + time_horizon + ", Time step: " + time_step + ", Curve time limit: " + curve_time_limit);
                for (int current_iteration = 0; current_iteration < n_iterations; current_iteration++) {
                    HashMap<String, double[]> probabilityOfNotBeingInfectedDueToPreviousContact = new HashMap<>();
                    for (int i = 0; i < n_subjects; i++) {
                        probabilityOfNotBeingInfectedDueToPreviousContact.put(String.valueOf(i + 1), new double[time_horizon]);
                        Arrays.fill(probabilityOfNotBeingInfectedDueToPreviousContact.get(String.valueOf(i + 1)), 1.0);
                    }
                    Tracks currentTracks = new Tracks(names, time_horizon);
                    tracksRecord[current_iteration] = currentTracks;
                    Tracks previousTracks = current_iteration > 0 ? tracksRecord[current_iteration - 1] : null;

                    for (Event event : events) {
                        int eventTime = (int) Math.round(event.time / time_step); // this is the event time scaled by the time step
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

                            Integer firstPositiveTestOffset = null;
                            ArrayList<Event> subjectSymptomsAndTests = symptomsAndTestsBySubject.get(involvedSubject);
                            if (subjectSymptomsAndTests != null) {
                                for (Event entry : subjectSymptomsAndTests) {
                                    if (entry.time < event.time) {
                                        continue;
                                    }

                                    int curveOffset = (int) Math.round((entry.time - event.time) / time_step);
                                    if (!hasObservationCurveOffset(observationCurves, curveOffset)) {
                                        continue;
                                    }

                                    if (entry.type.equals("Symptoms") && Boolean.TRUE.equals(entry.result)) {
                                        Pw_h_given_e_s_k_is_effective *= observationCurves.phi[curveOffset];
                                        prior *= priorsValues[0];
                                    } else if (entry.type.equals("Test")) {
                                        if (entry.result) {
                                            Pw_h_given_e_s_k_is_effective *= observationCurves.theta[curveOffset];
                                            prior *= priorsValues[1];
                                            if (firstPositiveTestOffset == null || curveOffset < firstPositiveTestOffset) {
                                                firstPositiveTestOffset = curveOffset;
                                            }
                                        } else {
                                            Pw_h_given_e_s_k_is_effective *= (1.0 - observationCurves.theta[curveOffset]);
                                            prior *= priorsValues[2];
                                        }
                                    }
                                }
                            }
                            double r_ext = Pw_h_given_e_s_k_is_effective * riskFactor_external / prior;

                            int offset_external = 0;
                            while (eventTime + offset_external < time_horizon) {
                                int sampleTime = eventTime + offset_external;

                                double[] probabilityTrack = probabilityOfNotBeingInfectedDueToPreviousContact.get(involvedSubject);
                                double[] currentTrack = currentTracks.getTrack(involvedSubject);
                                double previousValue = currentTrack[sampleTime];

                                double stpn_value;
                                // After the first positive test, the kernel follows the
                                // survival curve of the time from positive test to isolation.
                                stpn_value = computeKernelValue(
                                        stpnSolutionMap,
                                        offset_external,
                                        firstPositiveTestOffset,
                                        observationCurves.psiSurvival
                                );

                                double newValue = stpn_value * r_ext * probabilityTrack[sampleTime] + previousValue;

                                probabilityTrack[sampleTime] *= (1.0 - r_ext);
                                currentTrack[sampleTime] = newValue;
                                offset_external++;
                            }
                        } else if (current_iteration > 0 && event.type.equals("Internal")) {
                            String highestRiskSubject = null;
                            double highestRisk = 0.0;
                            String secondHighestRiskSubject = null;
                            double secondHighestRisk = 0.0;
                            
                            // To prevent ping-pong feedback loops at the exact same event timestamp,
                            // we must sample the risk from strictly *before* the event occurs.
                            int priorTime = Math.max(0, eventTime - 1);
                            
                            for (String subject : involvedSubjects) { // get the highest risk subject
                                double subjectRisk = previousTracks.getSample(subject, priorTime);
                                if (subjectRisk > highestRisk) {
                                    highestRisk = subjectRisk;
                                    highestRiskSubject = subject;
                                }
                            }
                            for (String subject : involvedSubjects) { // get the second highest risk subject.
                                double subjectRisk = previousTracks.getSample(subject, priorTime);
                                if (subjectRisk > secondHighestRisk && subjectRisk < highestRisk) {
                                    secondHighestRisk = subjectRisk;
                                    secondHighestRiskSubject = subject;
                                }
                            }
                            double riskFactor_internal = event.riskFactor;
                            HashMap<String, Double> r_int_map = new HashMap<>();
                            HashMap<String, Integer> firstPositiveTestOffsetMap = new HashMap<>();
                            for (String subject : involvedSubjects) {
                                double Pw_h_given_e_s_k_is_effective = 1.0;
                                double prior = 1.0;
                                firstPositiveTestOffsetMap.put(subject, null);

                                ArrayList<Event> subjectSymptomsAndTests = symptomsAndTestsBySubject.get(subject);
                                if (subjectSymptomsAndTests != null) {
                                    for (Event entry : subjectSymptomsAndTests) {
                                        if (entry.time < event.time) {
                                            continue;
                                        }
                                        int curveOffset = (int) Math.round((entry.time - event.time) / time_step);
                                        if (!hasObservationCurveOffset(observationCurves, curveOffset)) {
                                            continue;
                                        }

                                        if (entry.type.equals("Symptoms") && Boolean.TRUE.equals(entry.result)) {
                                            Pw_h_given_e_s_k_is_effective *= observationCurves.phi[curveOffset];
                                            prior *= priorsValues[0];
                                        } else if (entry.type.equals("Test")) {
                                            if (entry.result) {
                                                Pw_h_given_e_s_k_is_effective *= observationCurves.theta[curveOffset];
                                                prior *= priorsValues[1];
                                                Integer currentPositiveTestOffset = firstPositiveTestOffsetMap.get(subject);
                                                if (currentPositiveTestOffset == null || curveOffset < currentPositiveTestOffset) {
                                                    firstPositiveTestOffsetMap.put(subject, curveOffset);
                                                }
                                            } else {
                                                Pw_h_given_e_s_k_is_effective *= (1.0 - observationCurves.theta[curveOffset]);
                                                prior *= priorsValues[2];
                                            }
                                        }
                                    }
                                }
                                r_int_map.put(subject, Pw_h_given_e_s_k_is_effective * riskFactor_internal / prior);
                            }

                            int offset_internal = 0;
                            while (eventTime + offset_internal < time_horizon) {
                                for (String subject : involvedSubjects) {
                                    int sampleTime = eventTime + offset_internal;

                                    double[] probabilityTrack = probabilityOfNotBeingInfectedDueToPreviousContact.get(subject);
                                    double[] currentTrack = currentTracks.getTrack(subject);
                                    double previousValue = currentTrack[sampleTime];
                                    double q = subject.equals(highestRiskSubject) ? secondHighestRisk : highestRisk;

                                    // Retrieve the pre-calculated r_int
                                    double r_int = r_int_map.get(subject);

                                    double solutionValue = computeKernelValue(
                                            stpnSolutionMap,
                                            offset_internal,
                                            firstPositiveTestOffsetMap.get(subject),
                                            observationCurves.psiSurvival
                                    );

                                    double newValue = r_int * q * solutionValue * probabilityTrack[sampleTime] + previousValue;

                                    probabilityTrack[sampleTime] *= (1.0 - (q * r_int));
                                    currentTrack[sampleTime] = newValue;
                                }
                                offset_internal++;
                            }

                        }
                    }
                }


                Tracks tracks = new Tracks(names, time_horizon);
                for (int i = 0; i < n_subjects; i++) {
                    String subjectName = String.valueOf(i + 1);
                    double[] aggregateTrack = tracks.getTrack(subjectName);
                    double[][] iterationTracks = new double[n_iterations][];
                    for (int k = 0; k < n_iterations; k++) {
                        iterationTracks[k] = tracksRecord[k].getTrack(subjectName);
                    }
                    for (int j = 0; j < time_horizon; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < n_iterations; k++) {
                            sum += iterationTracks[k][j];
                        }
                        aggregateTrack[j] = sum;
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

        double coreAnalysisRuntimeSeconds = (System.nanoTime() - coreAnalysisStartedAt) / 1_000_000_000.0;
        System.out.printf(Locale.US, "__TIMING__ core_analysis_runtime_seconds=%.9f%n", coreAnalysisRuntimeSeconds);
    }
}
