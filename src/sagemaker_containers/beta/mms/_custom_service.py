# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import importlib

from sagemaker_containers.beta.framework import (modules, env)


class ModelHandler(object):
    def __init__(self):
        serving_env = env.ServingEnv()

        # import support code
        framework_support = serving_env.framework_module.split(':')[0]
        user_module = modules.import_module(serving_env.module_dir, serving_env.module_name)

        # download and pip install the user's inference script and extract transform_fn
        framework_support_module = importlib.import_module(framework_support)
        user_module_transformer = getattr(framework_support_module, 'user_module_transformer')(user_module,
                                                                                               serving_env.model_dir)

        self.module_transformer = user_module_transformer
        self.initialized = False

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self.module_transformer.initialize()
        self.initialized = True

    def transform(self, data, context):
        t_data = data[0].get('body')
        return self.module_transformer._transform_fn(self.module_transformer._model,
                                                     t_data.decode(),
                                                     context.get_response_content_type(0),
                                                     context.get_response_content_type(0))


_service = ModelHandler()


def transform(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    result = _service.transform(data, context).response
    return [result]
