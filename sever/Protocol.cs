using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Text.Json.Serialization;
using System.Text.Json.Nodes;

namespace sever
{
    public class ClientRequest
    {
        [JsonPropertyName("Command")]
        public string? Command { get; set; }

        [JsonPropertyName("Params")]
        public JsonNode? Params { get; set; }
    }

    public class ServerResponse
    {
        [JsonPropertyName("Status")]
        public string? Status { get; set; }

        [JsonPropertyName("Message")]
        public string? Message { get; set; }

        [JsonPropertyName("Result")]
        public JsonNode? Result { get; set; }
    }
}


// Auto Clicker Enhanced
// Copyright (C) <2025> <Dinh Khởi Minh>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.