// Pi SDK types — provided by the host runtime, not installed as a package
declare module '@earendil-works/pi-coding-agent' {
  export interface ExtensionAPI {
    registerTool(tool: {
      name: string;
      description: string;
      parameters: Record<string, any>;
      execute: (toolCallId: string, params: Record<string, any>) => Promise<any>;
    }): void;
    on(event: string, handler: (...args: any[]) => any): void;
    registerCommand(name: string, opts: { description: string; handler: () => any }): void;
    ui?: { notify?: (message: string) => void };
    systemPrompt?: { inject?: (text: string) => void };
  }
}
