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

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class Tracks {
    private final HashMap<String, double[]> tracks;
    private final String[] names;
    private final int nSamples;

    public Tracks(String[] names, int nSamples) {
        this.names = names;
        this.nSamples = nSamples;
        tracks = new HashMap<>();

        for (String name : names) {
            tracks.put(name, new double[nSamples]);
            Arrays.fill(tracks.get(name), 0.0);
        }
    }

    public void editTrack(String name, int time, Double value) {
        tracks.get(name)[time] = value;
    }

    public Double getSample(String name, int time) {
        return tracks.get(name)[time];
    }

    public double[] getTrack(String name) {
        return tracks.get(name);
    }

    public Tracks copy() {
        Tracks copy = new Tracks(Arrays.copyOf(this.names, this.names.length), this.nSamples);
        for (String name : names) {
            copy.tracks.put(name, Arrays.copyOf(this.tracks.get(name), this.nSamples));
        }
        for (String name : names) {
            if (copy.tracks.get(name) == this.tracks.get(name)) {
                throw new AssertionError("Copied track shares the original array");
            }
            if (!Arrays.equals(copy.tracks.get(name), this.tracks.get(name))) {
                throw new AssertionError("Copied track values differ from the original");
            }
        }
        return copy;
    }

    public JsonObject toJson() {
        Gson gson = new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();
        JsonObject jsonObject = new JsonObject();
        for (Map.Entry<String, double[]> entry : tracks.entrySet()) {
            jsonObject.add(entry.getKey(), gson.toJsonTree(entry.getValue()));
        }
        return jsonObject;
    }
}
