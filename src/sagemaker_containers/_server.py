# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from __future__ import absolute_import

import importlib
import os
import re
import signal
import subprocess

import pkg_resources

import sagemaker_containers
from sagemaker_containers import _env, _files, _logging
from sagemaker_containers._modules import import_module

logger = _logging.get_logger()

UNIX_SOCKET_BIND = 'unix:/tmp/gunicorn.sock'

nginx_config_file = os.path.join('/etc', 'sagemaker-nginx.conf')
nginx_config_template_file = pkg_resources.resource_filename(sagemaker_containers.__name__, '/etc/nginx.conf.template')

mms_config_file = os.path.join('/etc', 'sagemaker-mms.conf')
mms_config_template_file = pkg_resources.resource_filename(sagemaker_containers.__name__,
                                                           '/etc/config.properties.template')
mms_custom_service = pkg_resources.resource_filename(sagemaker_containers.__name__,
                                                     '/src/beta/mms/_custom_service.py:transform')


def _create_nginx_config(serving_env):
    template = _files.read_file(nginx_config_template_file)

    pattern = re.compile(r'%(\w+)%')
    template_values = {
        'NGINX_HTTP_PORT': serving_env.http_port
    }

    config = pattern.sub(lambda x: template_values[x.group(1)], template)

    logger.info('nginx config: \n%s\n', config)

    _files.write_file(nginx_config_file, config)


def _create_mms_config(serving_env):
    template = _files.read_file(mms_config_template_file)

    pattern = re.compile(r'%(\w+)%')
    template_values = {
        'INFERENCE_HTTP_PORT': serving_env.http_port
    }

    config = pattern.sub(lambda x: template_values[x.group(1)], template)

    logger.info('nginx config: \n%s\n', config)

    _files.write_file(mms_config_file, config)


def _add_sigterm_handler(nginx, gunicorn):
    def _terminate(signo, frame):
        if nginx:
            try:
                os.kill(nginx.pid, signal.SIGQUIT)
            except OSError:
                pass

        try:
            os.kill(gunicorn.pid, signal.SIGTERM)
        except OSError:
            pass

    signal.signal(signal.SIGTERM, _terminate)


def _add_sigterm_handler_temp(mms):
    def _terminate(signo, frame):
        if mms:
            try:
                os.kill(mms.pid, signal.SIGQUIT)
            except OSError:
                pass

    signal.signal(signal.SIGTERM, _terminate)


def start(module_app):
    env = _env.ServingEnv()
    gunicorn_bind_address = '0.0.0.0:{}'.format(env.http_port)

    nginx = None

    if env.use_nginx:
        gunicorn_bind_address = UNIX_SOCKET_BIND
        _create_nginx_config(env)
        nginx = subprocess.Popen(['nginx', '-c', nginx_config_file])

    gunicorn = subprocess.Popen(['gunicorn',
                                 '--timeout', str(env.model_server_timeout),
                                 '-k', 'gevent',
                                 '-b', gunicorn_bind_address,
                                 '--worker-connections', str(1000 * env.model_server_workers),
                                 '-w', str(env.model_server_workers),
                                 '--log-level', 'info',
                                 module_app])

    _add_sigterm_handler_temp(nginx, gunicorn)

    # wait for child processes. if either exit, so do we.
    pids = {c.pid for c in [nginx, gunicorn] if c}
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break


def start_mms():
    env = _env.ServingEnv()

    # import support code
    framework_support = env.framework_module.split(':')[0]

    # download and pip install the user's inference script and extract transform_fn
    # framework_support_module = importlib.import_module(framework_support)
    # user_module_transformer = getattr(framework_support_module, 'user_module_transformer')(user_module, env.model_dir)

    logger.info('archive')
    subprocess.call(['model-archiver',
                     '--model-name', framework_support,
                     '--handler', mms_custom_service,
                     '--model-path', env.model_dir])

    _create_mms_config(env)

    logger.info('start mms')
    mms = subprocess.Popen(['mxnet-model-server',
                            '--start',
                            '--mms-config', mms_config_file])

    _add_sigterm_handler_temp(mms)

    # wait for child processes. if either exit, so do we.
    pids = {c.pid for c in [mms] if c}
    while True:
        pid, _ = os.wait()
        if pid in pids:
            break


def next_safe_port(port_range, after=None):
    first_and_last_port = port_range.split('-')
    first_safe_port = int(first_and_last_port[0])
    last_safe_port = int(first_and_last_port[1])
    safe_port = first_safe_port
    if after:
        safe_port = int(after) + 1
        if safe_port < first_safe_port or safe_port > last_safe_port:
            raise ValueError(
                '{} is outside of the acceptable port range for SageMaker: {}'.format(safe_port, port_range))

    return str(safe_port)
