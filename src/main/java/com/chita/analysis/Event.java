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

public class Event {
    String type;
    String[] involvedSubjects;
    double time;
    Double riskFactor;
    Boolean result;

    public Event(String type, String[] involvedSubjects, double time, Double riskFactor, Boolean result) {
        this.type = type;
        this.involvedSubjects = involvedSubjects;
        this.time = time;
        this.riskFactor = riskFactor;
        this.result = result;
    }
}
