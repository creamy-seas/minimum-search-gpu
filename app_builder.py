from typing import Dict, List

import pinject


class AppBuilder:
    @classmethod
    def build(
        cls,
        param_dictionary: Dict,
        flux_list: List,
        logging_level: int,
        other_parameters: Dict = {"empty": "empty"},
    ):

        # Load the details into the InitDetails class
        BINDING_SPECS = [
            TwinQubitInitDetails(
                param_dictionary, flux_list, logging_level, other_parameters
            )
        ]

        OBJ_GRAPH = pinject.new_object_graph(binding_specs=BINDING_SPECS)

        APP = OBJ_GRAPH.provide(TwinQubit)

        return APP
