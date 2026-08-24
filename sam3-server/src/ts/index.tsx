import React, { useState } from "react";
import Input from "@modules/antd/lib/input";
import { type MLModel } from "cvat-core-wrapper";
import {
  APIWrapperEnterOptions,
  ComponentBuilder,
  PluginEntryPoint,
} from "components/plugins-entrypoint";

interface InteractorArguments {
  text_prompts?: string[];
}

interface SAM3ServerPlugin {
  name: string;
  description: string;
  cvat: {
    lambda: {
      call: {
        enter: (
          plugin: SAM3ServerPlugin,
          taskID: number,
          model: MLModel,
          args: InteractorArguments,
        ) => Promise<null | APIWrapperEnterOptions>;
      };
    };
  };
  data: {
    modelID: string;
    textPrompt: string;
  };
}

const sam3ServerPlugin: SAM3ServerPlugin = {
  name: "Segment Anything 3 Server",
  description: "Runs all SAM3 inference in the serverless function",
  cvat: {
    lambda: {
      call: {
        async enter(plugin, _taskID, model, args) {
          if (model.id === plugin.data.modelID) {
            Object.assign(args, {
              text_prompts: [plugin.data.textPrompt.trim() || "visual"],
            });
          }
          return null;
        },
      },
    },
  },
  data: {
    modelID: "pth-facebookresearch-sam3-server",
    textPrompt: "",
  },
};

function TextPromptInput(): JSX.Element {
  const [value, setValue] = useState(sam3ServerPlugin.data.textPrompt);
  return (
    <Input
      allowClear
      value={value}
      placeholder="Optional SAM3 text prompt"
      onKeyDown={(event) => event.stopPropagation()}
      onChange={(event) => {
        setValue(event.target.value);
        sam3ServerPlugin.data.textPrompt = event.target.value;
      }}
    />
  );
}

const builder: ComponentBuilder = ({ core, dispatch, actionCreators }) => {
  core.plugins.register(sam3ServerPlugin);
  dispatch(
    actionCreators.addUIComponent(
      "aiTools.interactors.extras",
      TextPromptInput,
      {
        weight: 11,
        shouldBeRendered: (_props, state: any) =>
          state.activeInteractor?.id === sam3ServerPlugin.data.modelID,
      },
    ),
  );

  return {
    name: sam3ServerPlugin.name,
    destructor: () => {
      dispatch(
        actionCreators.removeUIComponent(
          "aiTools.interactors.extras",
          TextPromptInput,
        ),
      );
    },
  };
};

function register(): void {
  if (Object.prototype.hasOwnProperty.call(window, "cvatUI")) {
    (
      window as any as { cvatUI: { registerComponent: PluginEntryPoint } }
    ).cvatUI.registerComponent(builder);
  }
}

window.addEventListener("plugins.ready", register, { once: true });
